name := "spark-template"

version := "0.2"

scalaVersion := "2.10.4"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.2.0"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.1.0"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.1.0"

resolvers += Resolver.sonatypeRepo("public")