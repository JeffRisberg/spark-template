package com.incra

import org.apache.log4j.{Level, Logger}

import scopt.OptionParser
import breeze.linalg.{DenseVector, squaredDistance}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
 * An example k-means app.
 */
object DenseKMeans {

  object InitializationMode extends Enumeration {
    type InitializationMode = Value
    val Random, Parallel = Value
  }

  import com.incra.DenseKMeans.InitializationMode._

  case class Params(
                     input: String = "./test-data/kmeans_test_data.txt",
                     k: Int = 3,
                     numIterations: Int = 10,
                     initializationMode: InitializationMode = Parallel)

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("DenseKMeans") {
      head("DenseKMeans: an example k-means app for dense data.")
      opt[Int]('k', "k")
        .text(s"number of clusters")
        .action((x, c) => c.copy(k = x))
      opt[Int]("numIterations")
        .text(s"number of iterations, default; ${defaultParams.numIterations}")
        .action((x, c) => c.copy(numIterations = x))
      opt[String]("initMode")
        .text(s"initialization mode (${InitializationMode.values.mkString(",")}), " +
        s"default: ${defaultParams.initializationMode}")
        .action((x, c) => c.copy(initializationMode = InitializationMode.withName(x)))
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"DenseKMeans with $params").setMaster("local")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    val examples = sc.textFile(params.input).map { line =>
      Vectors.dense(line.split(' ').map(_.toDouble))
    }.cache()

    val numExamples = examples.count()

    println(s"numExamples = $numExamples.")

    val initMode = params.initializationMode match {
      case Random => KMeans.RANDOM
      case Parallel => KMeans.K_MEANS_PARALLEL
    }

    val model = new KMeans()
      .setInitializationMode(initMode)
      .setK(params.k)
      .setMaxIterations(params.numIterations)
      .run(examples)

    model.clusterCenters.foreach { println(_) }

    val cost = model.computeCost(examples)
    println(s"Total cost = $cost.")

    // predict
    val point1: Vector = Vectors.dense(1.0, 1.1)
    val prediction1: Int = model.predict(point1)
    println(s"Point $point1 prediction $prediction1")

    val point2: Vector = Vectors.dense(4.0, 4.0)
    val prediction2: Int = model.predict(point2)
    println(s"Point $point2 prediction $prediction2")

    val point3: Vector = Vectors.dense(1.1, 4.95)
    val prediction3: Int = model.predict(point3)
    println(s"Point $point3 prediction $prediction3")

    sc.stop()
  }
}
