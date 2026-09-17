plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.aiobs"
    compileSdk {
        version = release(37)
    }

    // 锁定使用已安装且正常的 NDK
    ndkVersion = "26.1.10909125"

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    defaultConfig {
        applicationId = "com.example.aiobs"
        minSdk = 24
        targetSdk = 37
        versionCode = 1
        versionName = "11.14"

        testInstrumentationRunner =
            "androidx.test.runner.AndroidJUnitRunner"

        // 当前设备目标只保留 arm64
        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        // Native YUV converter 的 CMake 配置
        externalNativeBuild {
            cmake {
                cppFlags += listOf(
                    "-std=c++17",
                    "-O3"
                )
            }
        }
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    // CMake 工程位置：
    // app/src/main/native/CMakeLists.txt
    externalNativeBuild {
        cmake {
            path = file("src/main/native/CMakeLists.txt")
        }
    }
}

dependencies {
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx)
    implementation(libs.material)

    testImplementation(libs.junit)

    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)

    // 基础推流库
    implementation("com.github.pedroSG94.RootEncoder:library:2.5.0")

    // OpenCV
    implementation("org.opencv:opencv:4.13.0")


    // TensorFlow Lite
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.17.0")
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.17.0")

    // Qualcomm QNN
    implementation("com.qualcomm.qti:qnn-runtime:2.49.0")
    implementation("com.qualcomm.qti:qnn-litert-delegate:2.49.0")
}