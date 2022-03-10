import sbt.librarymanagement.ConflictWarning

// enablePlugins(JavaAppPackaging)

name := "akka-http-catboost"
organization := "leosmerling"
version := "1.0"
scalaVersion := "3.1.1"

conflictWarning := ConflictWarning.disable

scalacOptions := Seq("-unchecked", "-deprecation", "-encoding", "utf8")

libraryDependencies ++= {
  val akkaHttpV      = "10.2.9"
  val akkaV          = "2.6.18"
  val scalaTestV     = "3.2.11"
  val circeV         = "0.14.1"
  val akkaHttpCirceV = "1.39.2"

  Seq(
    "io.circe"          %% "circe-core" % circeV,
    "io.circe"          %% "circe-parser" % circeV,
    "io.circe"          %% "circe-generic" % circeV,
    "ch.qos.logback"    % "logback-classic" % "1.2.3",
    "org.scalatest"     %% "scalatest" % scalaTestV % Test,
  ) ++ Seq(
    "com.typesafe.akka" %% "akka-http" % akkaHttpV,
    "com.typesafe.akka" %% "akka-actor-typed" % akkaV,
    "com.typesafe.akka" %% "akka-stream" % akkaV,
    "de.heikoseeberger" %% "akka-http-circe" % akkaHttpCirceV,
    "com.typesafe.akka" %% "akka-testkit" % akkaV,
    "com.typesafe.akka" %% "akka-http-testkit" % akkaHttpV % Test,
    "com.typesafe.akka" %% "akka-http-testkit" % akkaHttpV % Test,
    "com.typesafe.akka" %% "akka-actor-testkit-typed" % akkaV % Test,
  ).map(_.cross(CrossVersion.for3Use2_13))
}

Revolver.settings
