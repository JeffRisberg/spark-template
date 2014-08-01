package com.incra.spark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater}
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
 * An example app for linear regression.
 */
object LinearRegression extends App {

  object RegType extends Enumeration {
    type RegType = Value
    val NONE, L1, L2 = Value
  }

  import com.incra.spark.LinearRegression.RegType._

  case class Params(
                     input: String = "./test-data/linear_regression_data.txt",
                     numIterations: Int = 100,
                     stepSize: Double = 1.0,
                     regType: RegType = L2,
                     regParam: Double = 0.1)

  val defaultParams = Params()

  val parser = new OptionParser[Params]("LinearRegression") {
    head("LinearRegression: an example app for linear regression.")
    opt[Int]("numIterations")
      .text("number of iterations")
      .action((x, c) => c.copy(numIterations = x))
    opt[Double]("stepSize")
      .text(s"initial step size, default: ${defaultParams.stepSize}")
      .action((x, c) => c.copy(stepSize = x))
    opt[String]("regType")
      .text(s"regularization type (${RegType.values.mkString(",")}), " +
      s"default: ${defaultParams.regType}")
      .action((x, c) => c.copy(regType = RegType.withName(x)))
    opt[Double]("regParam")
      .text(s"regularization parameter, default: ${defaultParams.regParam}")
    //opt[String]("<input>")
    //  .required()
    //  .text("input paths to labeled examples in LIBSVM format")
    //  .action((x, c) => c.copy(input = x))
    note(
      """
        |For example, the following command runs this app on a synthetic dataset:
        |
        | bin/spark-submit --class org.apache.spark.examples.mllib.LinearRegression \
        |  examples/target/scala-*/spark-examples-*.jar \
        |  data/mllib/linear_regression_data.txt
      """.stripMargin)
  }

  parser.parse(args, defaultParams).map { params =>
    run(params)
  } getOrElse {
    sys.exit(1)
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"LinearRegression with $params").setMaster("local")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    val examples = MLUtils.loadLibSVMFile(sc, params.input, multiclass = true).cache()

    val splits = examples.randomSplit(Array(0.8, 0.2))
    val training = splits(0).cache()
    val test = splits(1).cache()

    val numTraining = training.count()
    val numTest = test.count()
    println(s"Training: $numTraining, test: $numTest.")

    examples.unpersist(blocking = false)

    val updater = params.regType match {
      case NONE => new SimpleUpdater()
      case L1 => new L1Updater()
      case L2 => new SquaredL2Updater()
    }

    val algorithm = new LinearRegressionWithSGD()
    algorithm.optimizer
      .setNumIterations(params.numIterations)
      .setStepSize(params.stepSize)
      .setUpdater(updater)
      .setRegParam(params.regParam)

    val model = algorithm.run(training)

    val weights = model.weights.toArray.map(v => "%.4f".format(v)).mkString(",")
    println(s"Model intercept ${model.intercept}")
    println(s"Model weights $weights")

    val point1 = Vectors.dense(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val prediction1 = model.predict(point1)
    println(s"Point $point1 prediction $prediction1")

    val point2 = Vectors.dense(3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
    val prediction2 = model.predict(point2)
    println(s"Point $point2 prediction $prediction2")

    val prediction = model.predict(test.map(_.features))
    val predictionAndActual = prediction.zip(test.map(_.label))

    val loss = predictionAndActual.map { case (p, actual) =>
      val err = p - actual
      err * err
    }.reduce(_ + _)
    val rmse = math.sqrt(loss / numTest)

    println(s"Test RMSE = $rmse.")

    sc.stop()
  }
}