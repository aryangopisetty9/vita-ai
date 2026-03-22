allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    // Only redirect the app's build output.  Third-party Flutter plugin
    // subprojects (e.g. camera_android_camerax) have their source files in
    // the pub cache on a different drive (C:\Users\...\Pub\Cache).  Redirecting
    // their build output to E:\ causes javac to fail with "source and base files
    // have different roots" on Windows.  Leave plugin builds in their own
    // default location so both source and output stay on the same drive.
    if (project.name == "app") {
        val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
        project.layout.buildDirectory.value(newSubprojectBuildDir)
    }
}
subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
