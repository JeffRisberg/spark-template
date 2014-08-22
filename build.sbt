name := "spark_template"

version := "0.1"

scalaVersion := "2.10.4"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.2.0"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.0.2"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.0.2"

resolvers += Resolver.sonatypeRepo("public")