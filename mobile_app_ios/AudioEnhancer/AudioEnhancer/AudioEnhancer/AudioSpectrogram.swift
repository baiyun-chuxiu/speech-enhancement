import Foundation
import Accelerate

class AudioSpectrogram {
    let n_fft: Int
    let hop_length: Int
    let sampleRate: Float
    
    private let fftSetup: vDSP_DFT_Setup
    private let window: [Float]
    private let windowScale: Float
    
    init(n_fft: Int = 2048, hop_length: Int = 512, sampleRate: Float = 44100) {
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sampleRate = sampleRate
        self.fftSetup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(n_fft), vDSP_DFT_Direction.FORWARD)!
        
        var win = [Float](repeating: 0, count: n_fft)
        vDSP_hann_window(&win, vDSP_Length(n_fft), Int32(vDSP_HANN_DENORM))
        self.window = win
        
        var winSum: Float = 0
        vDSP_sve(window, 1, &winSum, vDSP_Length(n_fft))
        self.windowScale = winSum / Float(hop_length)
    }
    
    deinit {
        vDSP_DFT_DestroySetup(fftSetup)
    }
    
    // MARK: - STFT
    func stft(audioData: [Float]) -> ([[Float]], [[Float]]) {
        let padLen = n_fft / 2
        var paddedAudio = [Float](repeating: 0, count: padLen + audioData.count + padLen)
        paddedAudio.replaceSubrange(padLen..<padLen+audioData.count, with: audioData)
        
        let totalFrames = (paddedAudio.count - n_fft) / hop_length + 1
        var magnitudes = [[Float]]()
        magnitudes.reserveCapacity(totalFrames)
        var phases = [[Float]]()
        phases.reserveCapacity(totalFrames)
        
        var realPart = [Float](repeating: 0, count: n_fft)
        var imagPart = [Float](repeating: 0, count: n_fft)
        var outReal = [Float](repeating: 0, count: n_fft)
        var outImag = [Float](repeating: 0, count: n_fft)
        
        let binCount = n_fft / 2 + 1
        var mag = [Float](repeating: 0, count: binCount)
        var pha = [Float](repeating: 0, count: binCount)
        
        paddedAudio.withUnsafeBufferPointer { audioPtr in
            guard let audioBase = audioPtr.baseAddress else { return }
            for i in 0..<totalFrames {
                let start = i * hop_length
                if start + n_fft > paddedAudio.count { break }
                cblas_scopy(Int32(n_fft), audioBase + start, 1, &realPart, 1)
                vDSP_vmul(realPart, 1, window, 1, &realPart, 1, vDSP_Length(n_fft))
                vDSP_vclr(&imagPart, 1, vDSP_Length(n_fft))
                vDSP_DFT_Execute(fftSetup, &realPart, &imagPart, &outReal, &outImag)
                
                outReal.withUnsafeMutableBufferPointer { rBuf in
                    outImag.withUnsafeMutableBufferPointer { iBuf in
                        var complex = DSPSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
                        vDSP_zvabs(&complex, 1, &mag, 1, vDSP_Length(binCount))
                        vDSP_zvphas(&complex, 1, &pha, 1, vDSP_Length(binCount))
                    }
                }
                magnitudes.append(mag)
                phases.append(pha)
            }
        }
        return (magnitudes, phases)
    }
    
    // MARK: - Process
    func processToModelInput(magnitudes: [[Float]]) -> [[Float]] {
        if magnitudes.isEmpty { return [] }
        let rows = magnitudes.count
        let cols = magnitudes[0].count
        var allValues = magnitudes.flatMap { $0 }
        let count = vDSP_Length(allValues.count)
        
        var amin: Float = 1e-5; vDSP_vthr(allValues, 1, &amin, &allValues, 1, count)
        var n = Int32(count); vvlog10f(&allValues, allValues, &n)
        var mul: Float = 20.0; vDSP_vsmul(allValues, 1, &mul, &allValues, 1, count)
        var maxDb: Float = 0; vDSP_maxv(allValues, 1, &maxDb, count)
        var thr = maxDb - 80.0; vDSP_vthr(allValues, 1, &thr, &allValues, 1, count)
        var minDb: Float = 0; vDSP_minv(allValues, 1, &minDb, count)
        let range = maxDb - minDb + 1e-9
        var negMin = -minDb; vDSP_vsadd(allValues, 1, &negMin, &allValues, 1, count)
        var invRange = 1.0 / range; vDSP_vsmul(allValues, 1, &invRange, &allValues, 1, count)
        
        var result = [[Float]]()
        result.reserveCapacity(rows)
        for i in 0..<rows {
            result.append(Array(allValues[i*cols..<(i+1)*cols]))
        }
        return result
    }
    
    // MARK: - ISTFT
    func istft(magnitudes: [[Float]], phases: [[Float]]) -> [Float] {
        guard !magnitudes.isEmpty else { return [] }
        let numFrames = magnitudes.count
        let outputLen = (numFrames - 1) * hop_length + n_fft
        
        var outputSignal = [Float](repeating: 0, count: outputLen)
        var outputNorm = [Float](repeating: 0, count: outputLen)
        let inverseSetup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(n_fft), vDSP_DFT_Direction.INVERSE)!
        defer { vDSP_DFT_DestroySetup(inverseSetup) }
        
        var realPart = [Float](repeating: 0, count: n_fft)
        var imagPart = [Float](repeating: 0, count: n_fft)
        var outReal = [Float](repeating: 0, count: n_fft)
        var outImag = [Float](repeating: 0, count: n_fft)
        var windowedOut = [Float](repeating: 0, count: n_fft)
        var scale = 1.0 / Float(n_fft)
        
        outputSignal.withUnsafeMutableBufferPointer { outputBuf in
            guard let outputBase = outputBuf.baseAddress else { return }
            for i in 0..<numFrames {
                let mag = magnitudes[i]; let pha = phases[i]
                for k in 0..<mag.count {
                    realPart[k] = mag[k] * cos(pha[k])
                    imagPart[k] = mag[k] * sin(pha[k])
                }
                for k in 1..<(n_fft - mag.count + 1) {
                    let sym = n_fft - k
                    realPart[sym] = realPart[k]
                    imagPart[sym] = -imagPart[k]
                }
                vDSP_DFT_Execute(inverseSetup, &realPart, &imagPart, &outReal, &outImag)
                vDSP_vsmul(outReal, 1, &scale, &outReal, 1, vDSP_Length(n_fft))
                vDSP_vmul(outReal, 1, window, 1, &windowedOut, 1, vDSP_Length(n_fft))
                
                let start = i * hop_length
                if start + n_fft <= outputLen {
                    let ptr = outputBase + start
                    vDSP_vadd(ptr, 1, windowedOut, 1, ptr, 1, vDSP_Length(n_fft))
                    for j in 0..<n_fft { outputNorm[start + j] += window[j] * window[j] }
                } else {
                    for j in 0..<n_fft {
                        if start + j < outputLen {
                            outputBase[start + j] += windowedOut[j]
                            outputNorm[start + j] += window[j] * window[j]
                        }
                    }
                }
            }
        }
        for j in 0..<outputLen {
            if outputNorm[j] > 1e-6 { outputSignal[j] /= outputNorm[j] }
        }
        let pad = n_fft / 2
        return (outputSignal.count > 2 * pad) ? Array(outputSignal[pad..<(outputSignal.count - pad)]) : outputSignal
    }
    
    func denormalize(modelOutput: [[Float]], minDb: Float = -80.0, maxDb: Float = 0.0) -> [[Float]] {
        if modelOutput.isEmpty { return [] }
        let rows = modelOutput.count; let cols = modelOutput[0].count
        var allValues = modelOutput.flatMap { $0 }
        let count = vDSP_Length(allValues.count)
        
        var range = maxDb - minDb; vDSP_vsmul(allValues, 1, &range, &allValues, 1, count)
        var minV = minDb; vDSP_vsadd(allValues, 1, &minV, &allValues, 1, count)
        var div20: Float = 0.05; vDSP_vsmul(allValues, 1, &div20, &allValues, 1, count)
        for i in 0..<allValues.count { allValues[i] = pow(10.0, allValues[i]) }
        
        var result = [[Float]]()
        for i in 0..<rows { result.append(Array(allValues[i*cols..<(i+1)*cols])) }
        return result
    }
}
