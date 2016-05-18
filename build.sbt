name := "anomaly-detection"

version := "1.0"

scalaVersion := "2.10.5"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.6.1"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.6.1"

libraryDependencies += "com.holdenkarau" %% "spark-testing-base" % "1.6.1_0.3.3" // includes scalatest 2.2.1
libraryDependencies += "org.scalactic" %% "scalactic" % "2.2.1"


javaOptions ++= Seq("-Xms512M", "-Xmx2048M", "-XX:MaxPermSize=2048M", "-XX:+CMSClassUnloadingEnabled")
parallelExecution in Test := false