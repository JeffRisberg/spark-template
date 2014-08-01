package com.incra.spark

import org.apache.spark._

import scala.math.random

/** Computes an approximation to pi */
object SparkPi {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Spark Pi").setMaster("local")
    val spark = new SparkContext(conf)
    val slices = if (args.length > 0) args(0).toInt else 2
    val n = 100000 * slices

    val count = spark.parallelize(1 to n, slices).map { i =>
      val x = random * 2 - 1
      val y = random * 2 - 1
      if (x * x + y * y < 1) 1 else 0
    }.reduce(_ + _)

    println("Pi is roughly " + 4.0 * count / n)

    // block so we can look at http://localhost:4040
    Thread.sleep(1000 * 3600)
  }
}
