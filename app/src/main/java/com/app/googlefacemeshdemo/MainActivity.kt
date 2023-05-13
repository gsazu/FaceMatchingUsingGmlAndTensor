package com.app.googlefacemeshdemo

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.hardware.Camera
import android.hardware.Camera.CameraInfo
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.app.googlefacemeshdemo.databinding.ActivityMainBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.facemesh.FaceMeshDetection
import com.google.mlkit.vision.facemesh.FaceMeshDetectorOptions
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val PICK_IMAGE = 10010
    private var permissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.READ_EXTERNAL_STORAGE,
        Manifest.permission.WRITE_EXTERNAL_STORAGE
    )
    val highAccuracyOpts = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .build()

    // Real-time contour detection
    private val realTimeOpts = FaceDetectorOptions.Builder()
        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
        .build()
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService

    protected var tflite: Interpreter? = null
    private var imageSizeX = 0
    private var imageSizeY = 0

    private val IMAGE_MEAN = 0.0f
    private val IMAGE_STD = 1.0f

    var oribitmap: Bitmap? = null
    var cropped: Bitmap? = null

    var ori_embedding = Array(1) {
        FloatArray(
            128
        )
    }
    var test_embedding = Array(1) {
        FloatArray(
            128
        )
    }

    companion object {
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        checkPermission(permissions, this)
//        FirebaseApp.initializeApp(this)
        cameraExecutor = Executors.newSingleThreadExecutor()
        val cId = findFrontFacingCameraID()


        initComponents()
    }

    private fun initComponents() {
        try {
            tflite = loadmodelfile(this).let { Interpreter(it) }
        } catch (e: java.lang.Exception) {
            Log.e("csdcsdc", e.message.toString())
        }
        binding.btnMatch.setOnClickListener {
            val distance = calculate_distance(ori_embedding, test_embedding)
            Log.e("dashjkdashdkjhasd", "dsahdjkashdhkjsd$distance")
            if (distance < 4.0) binding.tvResult.text =
                "Result : Same Faces" else binding.tvResult.text =
                "Result : Different Faces"
        }

        binding.btnPickImage1.setOnClickListener {

            binding.rlCamera.visibility = View.VISIBLE
            binding.llMainView.visibility = View.GONE
            cameraExecutor = Executors.newSingleThreadExecutor()
            startCamera("original")
        }

        binding.btnPickImage2.setOnClickListener {
            binding.rlCamera.visibility = View.VISIBLE
            binding.llMainView.visibility = View.GONE
            cameraExecutor = Executors.newSingleThreadExecutor()
            startCamera("testing")
        }

    }

    private fun calculate_distance(
        ori_embedding: Array<FloatArray>,
        test_embedding: Array<FloatArray>,
    ): Double {
        var sum = 0.0
        for (i in 0..127) {
            sum += Math.pow((ori_embedding[0][i] - test_embedding[0][i]).toDouble(), 2.0)
        }
        return Math.sqrt(sum)
    }

    private fun loadImage(bitmap: Bitmap, inputImageBuffer: TensorImage): TensorImage {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap)

        // Creates processor for the TensorImage.
        val cropSize = Math.min(bitmap.width, bitmap.height)
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        val imageProcessor: ImageProcessor = ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(getPreprocessNormalizeOp())
            .build()
        return imageProcessor.process(inputImageBuffer)
    }

    @Throws(IOException::class)
    private fun loadmodelfile(activity: Activity): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd("Qfacenet.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startoffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength)
    }

    private fun getPreprocessNormalizeOp(): TensorOperator? {
        return NormalizeOp(IMAGE_MEAN, IMAGE_STD)
    }


    fun face_detector(image: InputImage, bitmap: Bitmap, imagetype: String, imageP: ImageProxy) {

        val defaultDetector = FaceMeshDetection.getClient()

//        val boundingBoxDetector = FaceMeshDetection.getClient(
//            FaceMeshDetectorOptions.Builder()
//                .setUseCase(UseCase.BOUNDING_BOX_ONLY)
//                .build()
//        )
        val detector: FaceDetector = FaceDetection.getClient(realTimeOpts)
        try {

            detector.process(image).addOnSuccessListener { faces ->
                if (faces.size == 1) {
                    cameraExecutor.shutdown()
                    binding.rlCamera.visibility = View.GONE
                    binding.llMainView.visibility = View.VISIBLE

//                    for(points in faces[0].allPoints){
//                        Log.d("facemesh", "cs  ${points.position}")
//                    }
//                        Log.d("facemesh", "cs  ${faces[0].allPoints.}")
                        Log.e("success", "Haiga aa  ${faces.size}")
                        val bounds: Rect = faces[0].boundingBox
//            cropped = Bitmap.createBitmap(
//                bitmap, bounds.left, bounds.top,
//                bitmap.width, bitmap.height
//            )
                        Log.d(
                            "rrr",
                            "scsdc  ${bounds.left}  ${bounds.right}    ${bounds.top}      ${bounds.bottom}    ${bitmap.width}  ${bitmap.height}"
                        )
//            cropped = Bitmap.createBitmap(
//                bitmap)
//            cropped = bitmap

                        cropped = Bitmap.createBitmap(
                            bitmap, bounds.left, bounds.top,
                            bounds.width(), bounds.height()
                        )
                        if (imagetype == "original") {
                            binding.ivFirst.setImageBitmap(cropped)
                        } else {
                            binding.ivSecond.setImageBitmap(cropped)
                        }

                        get_embaddings(cropped!!, imagetype)
                        Log.e("success", "Haiga aa")
                }
            }.addOnFailureListener { e ->
//        Toast.makeText(applicationContext, e.message, Toast.LENGTH_LONG).show()
//        imageP.close()
                Log.e("success", e.message.toString())
            }
        } catch (e: Exception) {
            Log.e("cdscc", e.message.toString())
            Log.e("success", e.message.toString())
        }
    }

    fun get_embaddings(bitmap: Bitmap, imagetype: String) {
        var inputImageBuffer: TensorImage
        val embedding = Array(1) {
            FloatArray(
                128
            )
        }
        val imageTensorIndex = 0
        val imageShape: IntArray =
            tflite!!.getInputTensor(imageTensorIndex).shape() // {1, height, width, 3}
        imageSizeY = imageShape[1]
        imageSizeX = imageShape[2]
        val imageDataType: DataType? = tflite!!.getInputTensor(imageTensorIndex).dataType()
        inputImageBuffer = TensorImage(imageDataType)
        inputImageBuffer = loadImage(bitmap, inputImageBuffer)
        tflite!!.run(inputImageBuffer.buffer, embedding)
        val org_image_code = ArrayList<Float>()
        if (imagetype == "original") {
            ori_embedding = embedding
            for (i in embedding.indices) {
                for (j in 0 until embedding[i].size) {
                    Log.e("dasnndasndasd", "dsadjbsnadasd" + embedding[i][j])
                    org_image_code.add(embedding[i][j])
                }
            }

//            for (int i=0; i<embedding.length;i++){
//                for (int j=0; j<embedding[i].length;j++){
//                    Log.e("dasnndasndasd","dsadjbsnadasd"+embedding[i][j]);
//                    org_image_code.add(embedding[i][j]);
//                }
//            }
            Log.e("dsandjsakbdkbas", "dsakldhkasjhda" + org_image_code)
        } else if (imagetype == "testing") {
            test_embedding = embedding
            for (i in embedding.indices) {
                for (j in embedding[i].indices) {
                    Log.e("dasnndasndasd", "dsadjbsnadasd" + embedding[i][j])
                }
            }
        }
    }

    fun getImageUri(inContext: Context, inImage: Bitmap?): Uri {
        val image = rotateBitmap(inImage, 270f)
        val bytes = ByteArrayOutputStream()
        image!!.compress(Bitmap.CompressFormat.JPEG, 100, bytes)
        val path = MediaStore.Images.Media.insertImage(
            inContext.contentResolver,
            image,
            "" + System.currentTimeMillis(),
            null
        )
        return Uri.parse(path)
    }

    fun rotateBitmap(original: Bitmap?, degrees: Float): Bitmap? {
        val width = original!!.width
        val height = original.height
        return if (width > height) {
            val matrix = Matrix()
            matrix.preRotate(degrees)
            Bitmap.createBitmap(original, 0, 0, width, height, matrix, true)
        } else {
            original
        }
    }

    fun rotateBoundBitmap(original: Bitmap?, rotation: Float): Bitmap? {

        val matrix = Matrix()
        matrix.preRotate(rotation)
        return Bitmap.createBitmap(original!!, 0, 0, original.width, original.height, matrix, false)

    }


    fun checkPermission(permission: Array<String>, actvity: Activity) {
        for (i in permission.indices) {
            if (ContextCompat.checkSelfPermission(
                    actvity,
                    permission[i]
                ) == PackageManager.PERMISSION_DENIED
            ) {
                // Requesting the permission
                ActivityCompat.requestPermissions(actvity, permission, 5000)
            } else {
//                    Toast.makeText(actvity, "Permission already granted", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK && requestCode == PICK_IMAGE) {
            val uri = data!!.data
            try {
                oribitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                binding.ivFirst.setImageBitmap(oribitmap)
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
    }

    private fun findFrontFacingCameraID(): Int {
        var cameraId = -1
        // Search for the front facing camera
        val numberOfCameras = Camera.getNumberOfCameras()
        Log.d("camera_id", "No of Cameras: $numberOfCameras")
        for (i in 0 until numberOfCameras) {
            val info = CameraInfo()
            Camera.getCameraInfo(i, info)

            if (info.facing == CameraInfo.CAMERA_FACING_BACK) {
                Log.d("camera_id", "Camera found with id $i")
                cameraId = i
                break
            }
        }
        return cameraId
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun startCamera(imagetype: String) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({

            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.cameraView.surfaceProvider)
                }
            imageCapture = ImageCapture.Builder()
                .build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(
                        cameraExecutor
                    ) { imageProxy ->
                        val mediaImage = imageProxy.image
//                        val imageBitmap = imageProxy.toBitmap()
                        val imageBitmap = rotateBoundBitmap(
                            imageProxy.toBitmap(),
                            imageProxy.imageInfo.rotationDegrees.toFloat()
                        )
                        if (mediaImage != null) {
                            Log.d(
                                "cameraRotation",
                                "scssc    ${imageProxy.imageInfo.rotationDegrees}"
                            )
                            val image = InputImage.fromMediaImage(
                                mediaImage,
                                imageProxy.imageInfo.rotationDegrees
                            )
                            face_detector(image, imageBitmap!!, imagetype, imageProxy)
                            imageProxy.close()
                        }
                    }
                }
            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {

                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer
                )

            } catch (exc: Exception) {
                Log.e("Error", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto(imagee: InputImage) {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return


        // Create time stamped name and MediaStore entry.
        val name = SimpleDateFormat(Companion.FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/CameraX-Image")
            }
        }

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions
            .Builder(
                contentResolver,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues
            )
            .build()

        // Set up image capture listener, which is triggered after photo has
        // been taken
        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e("Error", "Photo capture failed: ${exc.message}", exc)
                }

                override fun
                        onImageSaved(output: ImageCapture.OutputFileResults) {
                    val msg = "Photo capture succeeded: ${output.savedUri}"
                    val bitmap = getBitmapFromUri(output.savedUri!!)
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                    Log.d("Saved_image", msg)
                }
            }
        )
    }

    fun getBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            val parcelFileDescriptor =
                applicationContext.contentResolver.openFileDescriptor(uri, "r")
            val fileDescriptor = parcelFileDescriptor?.fileDescriptor
            val image = BitmapFactory.decodeFileDescriptor(fileDescriptor)
            parcelFileDescriptor?.close()
            image
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    override fun onBackPressed() {
        cameraExecutor.shutdown()
        if (binding.rlCamera.visibility == View.VISIBLE) {
            binding.rlCamera.visibility = View.GONE
            binding.llMainView.visibility = View.VISIBLE
        } else {
            finishAffinity()
        }
    }

    override fun onPause() {
        super.onPause()
        cameraExecutor.shutdown()
    }
}