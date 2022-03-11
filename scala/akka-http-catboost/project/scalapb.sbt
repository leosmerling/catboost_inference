addSbtPlugin("com.thesamet" % "sbt-protoc" % "1.0.3")

libraryDependencies ++= Seq(
    "com.thesamet.scalapb" %% "scalapb-runtime" % "0.11.9" % "protobuf",
    "com.thesamet.scalapb" %% "compilerplugin" % "0.11.9"
).map(_.cross(CrossVersion.for3Use2_13))
