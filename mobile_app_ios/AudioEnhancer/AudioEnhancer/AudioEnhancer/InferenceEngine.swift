import Foundation
import CoreML
import AVFoundation
import Combine
import Accelerate

class InferenceEngine: ObservableObject {
    @Published var status = "Please select folders"
    @Published var isProcessing = false
    @Published var processingTime = ""
    @Published var progress: Double = 0.0 // 0.0 to 1.0
    @Published var outputFolderURL: URL?
    
    private let lock = NSLock()
    
    private struct FileSet {
        let id: String
        let wav: URL
        let acc: URL
        let gyro: URL
    }
    
    // MARK: - Batch Processing Entry Point
    func runBatchPipeline(wavFolder: URL, accFolder: URL, gyroFolder: URL) {
        self.isProcessing = true
        self.status = "Scanning files..."
        self.progress = 0.0
        self.processingTime = ""
        self.outputFolderURL = nil
        
        let startTime = Date()
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            do {
                let _ = wavFolder.startAccessingSecurityScopedResource()
                let _ = accFolder.startAccessingSecurityScopedResource()
                let _ = gyroFolder.startAccessingSecurityScopedResource()
                
                defer {
                    wavFolder.stopAccessingSecurityScopedResource()
                    accFolder.stopAccessingSecurityScopedResource()
                    gyroFolder.stopAccessingSecurityScopedResource()
                }
                
                let fileSets = self.matchFiles(wavFolder: wavFolder, accFolder: accFolder, gyroFolder: gyroFolder)
                
                guard !fileSets.isEmpty else {
                    throw NSError(domain: "App", code: -1, userInfo: [NSLocalizedDescriptionKey: "No matching file sets found"])
                }
                
                DispatchQueue.main.async {
                    self.status = "Found \(fileSets.count) file sets, preparing parallel processing..."
                }
                
                let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                let outputDir = docsURL.appendingPathComponent("EnhancedResults_\(Int(Date().timeIntervalSince1970))")
                try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
                
                DispatchQueue.main.async { self.outputFolderURL = outputDir }
                
                DispatchQueue.main.async { self.status = "Loading AI models..." }
                
                let config = MLModelConfiguration()
                config.computeUnits = .all
                
                // 【核心修改】：初始化 1s 专属的 CoreML 模型类
                let genModel = try Generator_1s(configuration: config)
                let denoiseModel = try DenoiseNet_1s(configuration: config)
                
                let totalCount = fileSets.count
                var successCount = 0
                var processedCount = 0
                
                let concurrencyLimit = 6
                let semaphore = DispatchSemaphore(value: concurrencyLimit)
                let group = DispatchGroup()
                
                DispatchQueue.main.async { self.status = "Starting batch processing (Concurrency: \(concurrencyLimit))..." }
                
                for fileSet in fileSets {
                    group.enter()
                    semaphore.wait()
                    
                    DispatchQueue.global(qos: .userInteractive).async {
                        autoreleasepool {
                            do {
                                let outputFilename = "enhanced_segment_\(fileSet.id).wav"
                                let saveURL = outputDir.appendingPathComponent(outputFilename)
                                
                                try self.processSingleItem(
                                    wavURL: fileSet.wav,
                                    accURL: fileSet.acc,
                                    gyroURL: fileSet.gyro,
                                    saveURL: saveURL,
                                    genModel: genModel,
                                    denoiseModel: denoiseModel
                                )
                                
                                self.lock.lock()
                                successCount += 1
                                self.lock.unlock()
                            } catch {
                                print("❌ Failed to process segment_\(fileSet.id): \(error.localizedDescription)")
                            }
                            
                            self.lock.lock()
                            processedCount += 1
                            let currentCount = processedCount
                            self.lock.unlock()
                            
                            if currentCount % 2 == 0 || currentCount == totalCount {
                                let progressVal = Double(currentCount) / Double(totalCount)
                                DispatchQueue.main.async {
                                    self.progress = progressVal
                                    self.status = "Processing: \(currentCount)/\(totalCount) (Success: \(successCount))"
                                }
                            }
                        }
                        
                        semaphore.signal()
                        group.leave()
                    }
                }
                
                group.wait()
                
                let duration = Date().timeIntervalSince(startTime)
                let finalSuccess = successCount
                
                DispatchQueue.main.async {
                    self.processingTime = String(format: "⏱️ Total time: %.1f s (Avg %.2f s/item)", duration, duration / Double(totalCount))
                    self.status = "✅ Processing complete! Success: \(finalSuccess)/\(totalCount)"
                    self.progress = 1.0
                    self.isProcessing = false
                }
                
            } catch {
                DispatchQueue.main.async {
                    self.status = "Error: \(error.localizedDescription)"
                    self.isProcessing = false
                }
            }
        }
    }
    
    // MARK: - File Matching Logic
    private func matchFiles(wavFolder: URL, accFolder: URL, gyroFolder: URL) -> [FileSet] {
        let fm = FileManager.default
        
        func getFiles(in url: URL, ext: String) -> [String] {
            guard let urls = try? fm.contentsOfDirectory(at: url, includingPropertiesForKeys: nil) else { return [] }
            return urls.filter { $0.pathExtension.lowercased() == ext }.map { $0.lastPathComponent }
        }
        
        let wavFiles = getFiles(in: wavFolder, ext: "wav")
        let accFiles = getFiles(in: accFolder, ext: "txt")
        let gyroFiles = getFiles(in: gyroFolder, ext: "txt")
        
        // 【核心修改】：精准匹配 segment_x_z 结构提取 ID
        func extractID(from filename: String) -> String? {
            let name = (filename as NSString).deletingPathExtension // 剥离 .wav，留下 "segment_1_1"
            if name.hasPrefix("segment_") {
                return String(name.dropFirst("segment_".count))     // 剥离 "segment_"，留下 "1_1"
            }
            return name
        }
        
        var sets: [FileSet] = []
        for wavName in wavFiles {
            guard let id = extractID(from: wavName) else { continue }
            // 只要 txt 中包含 _1_1.txt 就匹配
            let matchedAcc = accFiles.first { $0.contains("_\(id).") }
            let matchedGyro = gyroFiles.first { $0.contains("_\(id).") }
            
            if let accName = matchedAcc, let gyroName = matchedGyro {
                let wUrl = wavFolder.appendingPathComponent(wavName)
                let aUrl = accFolder.appendingPathComponent(accName)
                let gUrl = gyroFolder.appendingPathComponent(gyroName)
                sets.append(FileSet(id: id, wav: wUrl, acc: aUrl, gyro: gUrl))
            }
        }
        // 根据数字大小对 file_id 进行初步排序（例如先按 x 排序）
        return sets.sorted { (Int($0.id.split(separator: "_").first ?? "0") ?? 0) < (Int($1.id.split(separator: "_").first ?? "0") ?? 0) }
    }
    
    // MARK: - Single Item Processing Logic
    // 【核心修改】：参数类型变更为 Generator_1s 和 DenoiseNet_1s
    private func processSingleItem(wavURL: URL, accURL: URL, gyroURL: URL, saveURL: URL, genModel: Generator_1s, denoiseModel: DenoiseNet_1s) throws {
        
        // 2. Process IMU Data
        guard let imuInput = DataProcessor.processIMUFiles(accURL: accURL, gyroURL: gyroURL) else {
            throw NSError(domain: "Pipeline", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid IMU data"])
        }
        
        // 3. Run Generator
        let genOutput = try genModel.prediction(imu_input: imuInput)
        guard let videoDisp = self.processGeneratorOutputFast(genOutput.video_disp_output) else {
            throw NSError(domain: "Pipeline", code: -3, userInfo: [NSLocalizedDescriptionKey: "Generator output error"])
        }
        
        // 4. Audio STFT
        let (audioData, sampleRate) = self.readWav(url: wavURL)
        if audioData.isEmpty {
            throw NSError(domain: "Pipeline", code: -2, userInfo: [NSLocalizedDescriptionKey: "Audio is empty"])
        }
        
        let stftTool = AudioSpectrogram(n_fft: 2048, hop_length: 512, sampleRate: Float(sampleRate))
        let (rawMagnitudes, rawPhases) = stftTool.stft(audioData: audioData)
        let normSpec = stftTool.processToModelInput(magnitudes: rawMagnitudes)
        
        // 【核心修改】：音频时间步从 431 缩小为 86
        let targetTimeSteps = 86
        let freqBins = 1025
        let fixedSpec = self.fixSpectrogramDimensions(normSpec, targetTime: targetTimeSteps, targetFreq: freqBins)
        
        // 5. Build CoreML Input
        let audioInputML = try MLMultiArray(shape: [1, 1, NSNumber(value: targetTimeSteps), NSNumber(value: freqBins)], dataType: .float32)
        let strides = audioInputML.strides.map { $0.intValue }
        let tStride = strides[2]
        let fStride = strides[3]
        
        try audioInputML.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in
            for t in 0..<targetTimeSteps {
                let rowStart = t * tStride
                let specRow = fixedSpec[t]
                for f in 0..<freqBins {
                    ptr[rowStart + f * fStride] = specRow[freqBins - 1 - f]
                }
            }
        }
        
        // 6. Run Denoise Model
        // 【核心修改】：使用新的 Input 类
        let denoiseInput = DenoiseNet_1sInput(audio_spectrogram: audioInputML, video_displacement: videoDisp)
        let denoiseOutput = try denoiseModel.prediction(input: denoiseInput)
        
        // 7. Post-processing
        let cleanedML = denoiseOutput.clean_spectrogram
        let cleanedSpec2D = try self.safeReadOutput(cleanedML, timeSteps: targetTimeSteps, freqBins: freqBins)
        let linearMag = stftTool.denormalize(modelOutput: cleanedSpec2D, minDb: -90.0, maxDb: 0.0)
        let alignedPhases = self.alignPhases(rawPhases, targetTime: targetTimeSteps, freqBins: freqBins)
        var finalAudio = stftTool.istft(magnitudes: linearMag, phases: alignedPhases)
        
        // Automatic Gain Control
        var maxAmp: Float = 0
        vDSP_maxmgv(finalAudio, 1, &maxAmp, vDSP_Length(finalAudio.count))
        if maxAmp > 0 {
            let targetPeak: Float = 0.95
            var scale = targetPeak / maxAmp
            vDSP_vsmul(finalAudio, 1, &scale, &finalAudio, 1, vDSP_Length(finalAudio.count))
        }
        
        // 8. Save Results
        self.saveWav(audioData: finalAudio, sampleRate: sampleRate, url: saveURL)
    }
    
    // MARK: - Helper Functions
    
    private func safeReadOutput(_ mlArray: MLMultiArray, timeSteps: Int, freqBins: Int) throws -> [[Float]] {
        var result = [[Float]](repeating: [Float](repeating: 0, count: freqBins), count: timeSteps)
        let strides = mlArray.strides.map { $0.intValue }
        let tStride = strides[2]
        let fStride = strides[3]
        
        if mlArray.dataType == .float32 {
            try mlArray.withUnsafeBufferPointer(ofType: Float.self) { ptr in
                for t in 0..<timeSteps {
                    let rowStart = t * tStride
                    for f in 0..<freqBins {
                        let val = ptr[rowStart + f * fStride]
                        result[t][freqBins - 1 - f] = val
                    }
                }
            }
        } else {
            for t in 0..<timeSteps {
                for f in 0..<freqBins {
                    let idx = [0, 0, NSNumber(value: t), NSNumber(value: f)] as [NSNumber]
                    result[t][freqBins - 1 - f] = mlArray[idx].floatValue
                }
            }
        }
        return result
    }
    
    private func processGeneratorOutputFast(_ rawOutput: MLMultiArray) -> MLMultiArray? {
        // 【核心修改】：依据神经网络感受野，1s Generator 原始输出为 32x48，需插值到 30x40
        let srcH = 32; let srcW = 48
        let dstH = 30; let dstW = 40
        do {
            // 【核心修改】：目标 MLMultiArray 大小改为 30x40
            let result = try MLMultiArray(shape: [1, 1, 30, 40], dataType: .float32)
            let rawStrides = rawOutput.strides.map { $0.intValue }
            let resStrides = result.strides.map { $0.intValue }
            let rawYStride = rawStrides[2]
            let rawXStride = rawStrides[3]
            let resYStride = resStrides[2]
            let resXStride = resStrides[3]
            
            if rawOutput.dataType == .float32 {
                try rawOutput.withUnsafeBufferPointer(ofType: Float.self) { rawPtr in
                    try result.withUnsafeMutableBufferPointer(ofType: Float.self) { resPtr, _ in
                        for y in 0..<dstH {
                            let invertedY = dstH - 1 - y
                            let srcY = Double(invertedY) * Double(srcH) / Double(dstH)
                            for x in 0..<dstW {
                                let srcX = Double(x) * Double(srcW) / Double(dstW)
                                let y1 = Int(floor(srcY)); let y2 = min(y1 + 1, srcH - 1)
                                let x1 = Int(floor(srcX)); let x2 = min(x1 + 1, srcW - 1)
                                let dy = Float(srcY - Double(y1))
                                let dx = Float(srcX - Double(x1))
                                let v11 = rawPtr[y1 * rawYStride + x1 * rawXStride]
                                let v12 = rawPtr[y1 * rawYStride + x2 * rawXStride]
                                let v21 = rawPtr[y2 * rawYStride + x1 * rawXStride]
                                let v22 = rawPtr[y2 * rawYStride + x2 * rawXStride]
                                let val = (v11 * (1 - dx) + v12 * dx) * (1 - dy) + (v21 * (1 - dx) + v22 * dx) * dy
                                resPtr[y * resYStride + x * resXStride] = (val + 1.0) / 2.0
                            }
                        }
                    }
                }
            } else {
                for y in 0..<dstH {
                    let invertedY = dstH - 1 - y
                    let srcY = Double(invertedY) * Double(srcH) / Double(dstH)
                    for x in 0..<dstW {
                        let srcX = Double(x) * Double(srcW) / Double(dstW)
                        let y1 = Int(floor(srcY)); let y2 = min(y1 + 1, srcH - 1)
                        let x1 = Int(floor(srcX)); let x2 = min(x1 + 1, srcW - 1)
                        let dy = srcY - Double(y1); let dx = srcX - Double(x1)
                        let v11 = rawOutput[[0, 0, NSNumber(value: y1), NSNumber(value: x1)]].doubleValue
                        let v12 = rawOutput[[0, 0, NSNumber(value: y1), NSNumber(value: x2)]].doubleValue
                        let v21 = rawOutput[[0, 0, NSNumber(value: y2), NSNumber(value: x1)]].doubleValue
                        let v22 = rawOutput[[0, 0, NSNumber(value: y2), NSNumber(value: x2)]].doubleValue
                        let val = (v11 * (1 - dx) + v12 * dx) * (1 - dy) + (v21 * (1 - dx) + v22 * dx) * dy
                        result[[0, 0, NSNumber(value: y), NSNumber(value: x)]] = NSNumber(value: (val + 1.0) / 2.0)
                    }
                }
            }
            return result
        } catch { return nil }
    }
    
    private func fixSpectrogramDimensions(_ spec: [[Float]], targetTime: Int, targetFreq: Int) -> [[Float]] {
        var result = spec
        if result.count > targetTime { result = Array(result.prefix(targetTime)) }
        else if result.count < targetTime {
            let padding = targetTime - result.count
            let empty = [Float](repeating: 0.0, count: targetFreq)
            for _ in 0..<padding { result.append(empty) }
        }
        return result
    }
    
    private func alignPhases(_ phases: [[Float]], targetTime: Int, freqBins: Int) -> [[Float]] {
        var result = phases
        if result.count > targetTime { result = Array(result.prefix(targetTime)) }
        else if result.count < targetTime {
            let padding = targetTime - result.count
            let empty = [Float](repeating: 0.0, count: freqBins)
            for _ in 0..<padding { result.append(empty) }
        }
        return result
    }
    
    private func readWav(url: URL) -> ([Float], Int) {
        let secured = url.startAccessingSecurityScopedResource()
        defer { if secured { url.stopAccessingSecurityScopedResource() } }
        
        guard let file = try? AVAudioFile(forReading: url),
              let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: file.fileFormat.sampleRate, channels: 1, interleaved: false),
              let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else {
            return ([], 44100)
        }
        
        try? file.read(into: buf)
        let data = Array(UnsafeBufferPointer(start: buf.floatChannelData?[0], count: Int(buf.frameLength)))
        return (data, Int(file.fileFormat.sampleRate))
    }
    
    private func saveWav(audioData: [Float], sampleRate: Int, url: URL) {
        if FileManager.default.fileExists(atPath: url.path) { try? FileManager.default.removeItem(at: url) }
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
        guard let file = try? AVAudioFile(forWriting: url, settings: format.settings),
              let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(audioData.count)) else { return }
        buf.frameLength = AVAudioFrameCount(audioData.count)
        if let ptr = buf.floatChannelData?[0] {
            for (i, val) in audioData.enumerated() { ptr[i] = val }
        }
        try? file.write(from: buf)
    }
}
