name := "spark-template"

version := "0.2"

scalaVersion := "2.10.5"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.2.0"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.3.1"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.3.1"

resolvers += Resolver.sonatypeRepo("public")