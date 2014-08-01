package com.incra.spark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
 * Time Series Classification example
 */
object TimeSeriesClassification {

  case class Params(
                     input: String = "./test-data/time_series_classification_data.csv",
                     numIterations: Int = 100,
                     stepSize: Double = 1.0,
                     regParam: Double = 0.1)

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("TimeSeriesClassification") {
      head("TimeSeriesClassification: detecting anomalies in time series using svm-based classification.")
      opt[Int]("numIterations")
        .text("number of iterations")
        .action((x, c) => c.copy(numIterations = x))
      opt[Double]("stepSize")
        .text(s"initial step size, default: ${defaultParams.stepSize}")
        .action((x, c) => c.copy(stepSize = x))
      opt[Double]("regParam")
        .text(s"regularization parameter, default: ${defaultParams.regParam}")
      //arg[String]("<input>")
      //  .required()
      //  .text("input paths to labeled examples in LIBSVM format")
      //  .action((x, c) => c.copy(input = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"Time Series Classification with $params").setMaster("local")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    // Read raw data
    val data = sc.textFile(params.input).map { line =>
      val parts = line.split(',')
      parts.map(_.toDouble)
    }

    // Set up Labeled Points
    val examples = data.map { row => {
      val label = row.head
      var points = row.drop(1)

      // Normalize
      val n = points.length
      val sumY = points.sum
      val sumX2 = (n - 1) * (n - 1) * (n - 1) / 3 + (n - 1) * (n - 1) / 2 + (n - 1) / 6
      val sumY2 = points.foldLeft(0.0) { (a, y) => a + y * y}

      val ybar = sumY / n
      var xbar = (n - 1) / 2.0
      val stdev = Math.sqrt(n * sumY2 - sumY * sumY) / n

      val normalizedPoints = points.map(y => if (stdev <= 0) (y - ybar) else (y - ybar) / stdev)

      // Feature extract
      var xxbar = 0.0
      var xybar = 0.0
      var x = 0
      points.foreach { y =>
        xxbar += (x - xbar) * (x - xbar)
        xybar += (x - xbar) * (y - ybar)
        x = x + 1
      }
      val slope = xybar / xxbar

      var largestDelta = normalizedPoints.head
      var oneSigmaOutliers = 0
      var twoSigmaOutliers = 0
      var threeSigmaOutliers = 0
      var priorValue = 0.0

      normalizedPoints.foreach { y =>
        if (Math.abs(y - priorValue) > largestDelta) largestDelta = Math.abs(y - priorValue)
        if (Math.abs(y) > 3) threeSigmaOutliers = threeSigmaOutliers + 1
        if (Math.abs(y) > 2) twoSigmaOutliers = twoSigmaOutliers + 1
        if (Math.abs(y) > 1) oneSigmaOutliers = oneSigmaOutliers + 1
        priorValue = y
      }
      println(s"yBar $ybar stddev $stdev slope $slope, largestDelta $largestDelta, oneSigmaOutliers $oneSigmaOutliers")

      val numFeatures = 7
      var indices = Array(0, 1, 2, 3, 4, 5, 6)
      var values = Array(ybar, stdev, slope, largestDelta, oneSigmaOutliers, twoSigmaOutliers, threeSigmaOutliers)

      LabeledPoint(label, Vectors.sparse(numFeatures, indices, values))
    }
    }.cache()

    val splits = examples.randomSplit(Array(0.8, 0.2))
    val training = splits(0).cache()
    val test = splits(1).cache()

    val numTraining = training.count()
    val numTest = test.count()
    println(s"Training: $numTraining, test: $numTest.")

    examples.unpersist(blocking = false)

    val updater = new L1Updater()

    val algorithm = new SVMWithSGD()
    algorithm.optimizer
      .setNumIterations(params.numIterations)
      .setStepSize(params.stepSize)
      .setUpdater(updater)
      .setRegParam(params.regParam)
    val model = algorithm.run(training).clearThreshold()
    println("Training done")

    val prediction = model.predict(test.map(_.features))
    val predictionAndLabel = prediction.zip(test.map(_.label))

    predictionAndLabel.foreach { x =>
      val result = if (x._1 > 0) 1.0 else 0.0
      val isCorrect = (result == x._2)

      println(s" test ${x._1} $result ${x._2} $isCorrect")
    }

    val accuracyA = predictionAndLabel.filter(x => (if (x._1 > 0) 1.0 else 0.0) == x._2).count().toDouble / numTest
    println(s"Test accuracy = ${accuracyA * 100.0} %.")

    val metrics = new BinaryClassificationMetrics(predictionAndLabel)

    println(s"Test areaUnderPR = ${metrics.areaUnderPR()}.")
    println(s"Test areaUnderROC = ${metrics.areaUnderROC()}.")

    sc.stop()
  }
}
