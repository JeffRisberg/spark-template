package com.incra.spark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
 * SVM Classification basic example
 */
object SVMClassification {

  case class Params(
                     input: String = "./test-data/svm_classification_data.txt",
                     numIterations: Int = 100,
                     stepSize: Double = 1.0,
                     regParam: Double = 0.1)

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("SVMClassification") {
      head("SVMClassification: an example app for svm classification.")
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
    val conf = new SparkConf().setAppName(s"SVM Classification with $params").setMaster("local")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    val examples = MLUtils.loadLibSVMFile(sc, params.input).cache()

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

      //(s" test ${x._1} $result ${x._2} $isCorrect")
    }

    val accuracyA = predictionAndLabel.filter(x => (if (x._1 > 0) 1.0 else 0.0) == x._2).count().toDouble / numTest
    println(s"Test accuracy = ${accuracyA * 100.0} %.")

    val metrics = new BinaryClassificationMetrics(predictionAndLabel)

    println(s"Test areaUnderPR = ${metrics.areaUnderPR()}.")
    println(s"Test areaUnderROC = ${metrics.areaUnderROC()}.")

    sc.stop()
  }
}
