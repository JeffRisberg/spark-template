package com.incra

import org.apache.spark.{SparkConf, SparkContext}

object Helper {

  def getSparkContext(args: Array[String]) = {
    assert(args.length == 0 || args.length == 2 || args.length == 3)
    val cores = Runtime.getRuntime.availableProcessors()
    val parallelism = if (args.length == 3) args(2) else s"${cores * 3}"
    val master = if (args.length > 0) args(0) else s"local[$cores]"
    val JARS = if (args.length > 0) Seq(args(1)) else Seq.empty
    var conf = new SparkConf()
      .setMaster(master)
      .setAppName("Spark Template Job")
      .set("spark.executor.memory", "1g")
      .set("spark.default.parallelism", parallelism)
      .setJars(JARS)
    if (System.getenv("SPARK_HOME") != null) {
      conf = conf.setSparkHome(System.getenv("SPARK_HOME"))
    }
    new SparkContext(conf)
  }

}
