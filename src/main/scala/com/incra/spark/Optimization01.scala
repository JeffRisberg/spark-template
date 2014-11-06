package com.incra.spark

import java.util

import breeze.linalg.{diff}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.optimization._
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import breeze.linalg.{axpy => brzAxpy}

import scala.collection.mutable.ListBuffer


/**
 * first Optimization Example
 */
object Optimization01 {

  case class Params(
                     numIterations: Int = 15,
                     numPoints: Int = 5,
                     stepSize: Double = 0.15,
                     regParam: Double = 0.1,
                     miniBatchFraction: Double = 1.0)

  class LeastSquaresGradient2 extends LeastSquaresGradient {
    override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {

      val x1 = weights.toArray(0)
      val x2 = weights.toArray(1)
      val value = (x1 - 2.0) * (x1 - 2.0) + (x2 - 4.0) * (x2 - 4.0)
      System.out.println(s"compute1:  $x1 $x2 $value")

      super.compute(data, label, weights)
    }

    override def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double = {

      val x1 = weights.toArray(0)
      val x2 = weights.toArray(1)
      val value = (x1 - 2.0) * (x1 - 2.0) + (x2 - 4.0) * (x2 - 4.0)
      System.out.println(s"compute1:  $x1 $x2 $value")

      super.compute(data, label, weights, cumGradient)
    }
  }

  class ExampleGradient1 extends Gradient {
    override def compute(data: linalg.Vector, label: Double, weights: linalg.Vector): (linalg.Vector, Double) = {

      val x1 = weights.toArray(0)
      val x2 = weights.toArray(1)
      val value = (x1 - 2.0) * (x1 - 2.0) + (x2 - 4.0) * (x2 - 4.0)
      System.out.println(s"compute1:  $x1 $x2 $value")

      val gradValues: Array[Double] = Array[Double](2.0 * (x1 - 2.0), 2.0 * (x2 - 4.0))
      val grad: linalg.Vector = new linalg.DenseVector(gradValues)
      val loss: Double = value - label

      (grad, loss)
    }

    override def compute(data: linalg.Vector, label: Double, weights: linalg.Vector, cumGradient: linalg.Vector): Double = {
      val x1 = weights.toArray(0)
      val x2 = weights.toArray(1)
      val value = (x1 - 2.0) * (x1 - 2.0) + (x2 - 4.0) * (x2 - 4.0)
      System.out.println(s"compute2: $x1 $x2 $value")

      val gradValues: Array[Double] = Array[Double](2.0 * (x1 - 2.0), 2.0 * (x2 - 4.0))
      System.out.println(s"   grad ${gradValues(0)} ${gradValues(1)}")

      cumGradient.toArray(0) = cumGradient.toArray(0) + gradValues(0)
      cumGradient.toArray(1) = cumGradient.toArray(1) + gradValues(1)

      val loss: Double = value - label

      loss
    }
  }

  class ExampleGradient2 extends Gradient {
    override def compute(data: linalg.Vector, label: Double, weights: linalg.Vector): (linalg.Vector, Double) = {

      val data1 = data.toArray(0)
      val weight1 = weights.toArray(0)
      val weight2 = weights.toArray(1)
      val value = weight1 + weight2 * data1
      val error = value - label
      System.out.println(s"compute1:  $weight1 $weight2 $value $error")

      val gradValues: Array[Double] = Array[Double](error, error * weight1)
      val grad: linalg.Vector = new linalg.DenseVector(gradValues)

      (grad, error * error)
    }

    override def compute(data: linalg.Vector, label: Double, weights: linalg.Vector, cumGradient: linalg.Vector): Double = {
      val data1 = data.toArray(0)
      val weight1 = weights.toArray(0)
      val weight2 = weights.toArray(1)
      val value = weight1 + weight2 * data1
      val error = value - label
      System.out.println(s"compute1:  $weight1 $weight2 $value $error")

      val gradValues: Array[Double] = Array[Double](error, error * weight1)
      System.out.println(s"   grad ${gradValues(0)} ${gradValues(1)}")

      cumGradient.toArray(0) = cumGradient.toArray(0) + gradValues(0)
      cumGradient.toArray(1) = cumGradient.toArray(1) + gradValues(1)

      error * error
    }
  }

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("Optimization01") {
      head("Optimization01: an example.")
      opt[Int]("numIterations")
        .text("number of iterations")
        .action((x, c) => c.copy(numIterations = x))
      opt[Int]("numPoints")
        .text("number of datapoints")
        .action((x, c) => c.copy(numPoints = x))
      opt[Double]("stepSize")
        .text(s"initial step size, default: ${defaultParams.stepSize}")
        .action((x, c) => c.copy(stepSize = x))
      opt[Double]("regParam")
        .text(s"regularization parameter, default: ${defaultParams.regParam}")
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"Optimization with $params").setMaster("local")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    val updater = new SimpleUpdater()

    val initialVector = Vectors.dense(0.0, 0.0)

    val gradient = new ExampleGradient2()

    var dataList = ListBuffer[(Double, Vector)]()
    for (i <- Range(0, params.numPoints)) {
      val target = (i + 1).toDouble
      val noise = Math.random() / 10.0 - 0.05

      val data = (target, Vectors.dense(1.0, i.toDouble + noise))
      dataList += data
    }

    val distData = sc.parallelize(dataList)

    val result = GradientDescent.runMiniBatchSGD(
      distData,
      gradient,
      updater,
      params.stepSize,
      params.numIterations,
      params.regParam, params.miniBatchFraction,
      initialVector)

    System.out.println(result)

    sc.stop()
  }
}
