import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject var engine = InferenceEngine()
    
    // Changed to store folder URLs
    @State private var wavFolderURL: URL?
    @State private var accFolderURL: URL?
    @State private var gyroFolderURL: URL?
    
    @State private var showWavPicker = false
    @State private var showAccPicker = false
    @State private var showGyroPicker = false
    
    var body: some View {
        VStack(spacing: 30) {
            Text("Batch Audio Enhancer App")
                .font(.title)
                .bold()
            
            Text("Please select folders containing the corresponding data")
                .font(.subheadline)
                .foregroundColor(.gray)
            
            // --- Folder Selection Area ---
            Group {
                FileSelectButton(title: "Select Audio Folder (wav)", selectedURL: wavFolderURL) {
                    showWavPicker = true
                }
                .fileImporter(isPresented: $showWavPicker, allowedContentTypes: [.folder]) { result in
                    if let url = try? result.get() { self.wavFolderURL = url }
                }
                
                FileSelectButton(title: "Select Accelerometer Folder (acc)", selectedURL: accFolderURL) {
                    showAccPicker = true
                }
                .fileImporter(isPresented: $showAccPicker, allowedContentTypes: [.folder]) { result in
                    if let url = try? result.get() { self.accFolderURL = url }
                }
                
                FileSelectButton(title: "Select Gyroscope Folder (gyro)", selectedURL: gyroFolderURL) {
                    showGyroPicker = true
                }
                .fileImporter(isPresented: $showGyroPicker, allowedContentTypes: [.folder]) { result in
                    if let url = try? result.get() { self.gyroFolderURL = url }
                }
            }
            
            Divider()
            
            // --- Run and Status ---
            ScrollView {
                VStack(spacing: 8) {
                    Text(engine.status)
                        .multilineTextAlignment(.center)
                        .foregroundColor(engine.status.contains("Error") ? .red : .gray)
                    
                    if !engine.processingTime.isEmpty {
                        Text(engine.processingTime)
                            .font(.headline)
                            .foregroundColor(.green)
                    }
                }
            }
            .frame(height: 100) // Limit height to prevent overly long logs
            
            if engine.isProcessing {
                VStack {
                    ProgressView(value: engine.progress, total: 1.0)
                        .progressViewStyle(LinearProgressViewStyle())
                        .padding(.horizontal)
                    Text(String(format: "Progress: %.0f%%", engine.progress * 100))
                        .font(.caption)
                }
            } else {
                Button(action: {
                    if let w = wavFolderURL, let a = accFolderURL, let g = gyroFolderURL {
                        // Start batch processing
                        engine.runBatchPipeline(wavFolder: w, accFolder: a, gyroFolder: g)
                    } else {
                        engine.status = "Please select all folders first"
                    }
                }) {
                    Text("Start Batch Processing")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(width: 200, height: 50)
                        .background((wavFolderURL != nil && accFolderURL != nil && gyroFolderURL != nil) ? Color.blue : Color.gray)
                        .cornerRadius(10)
                }
                .disabled(wavFolderURL == nil || accFolderURL == nil || gyroFolderURL == nil)
            }
            
            // Open output folder button
            if !engine.isProcessing && engine.outputFolderURL != nil {
                if !engine.isProcessing, let url = engine.outputFolderURL {
                    ShareLink(item: url) {
                        HStack {
                            Image(systemName: "square.and.arrow.up")
                            Text("Export Results Folder")
                        }
                        .font(.headline)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                }
            }
        }
        .padding()
    }
}

struct FileSelectButton: View {
    var title: String
    var selectedURL: URL?
    var action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: selectedURL == nil ? "folder.badge.plus" : "folder.fill")
                    .foregroundColor(selectedURL == nil ? .blue : .green)
                
                VStack(alignment: .leading) {
                    Text(title)
                        .foregroundColor(.primary)
                    if let url = selectedURL {
                        Text(url.lastPathComponent)
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                }
                Spacer()
                
                if selectedURL != nil {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                }
            }
            .padding()
            .background(Color(UIColor.secondarySystemBackground))
            .cornerRadius(8)
        }
    }
}
