package com.incra.sberbank

import au.com.bytecode.opencsv.CSVParser
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater}
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * An example app for linear regression.
  */
object PredictSales extends App {

  object RegType extends Enumeration {
    type RegType = Value
    val NONE, L1, L2 = Value
  }

  import PredictSales.RegType._

  case class Params(
                     numIterations: Int = 100,
                     stepSize: Double = 0.0010,
                     regType: RegType = L2,
                     regParam: Double = 0.1)

  val defaultParams = Params()

  def mLine(line: String) = {
    val parser = new CSVParser(',')
    parser.parseLine(line)
  }

  var priceDocIndex = 0
  var fullSqIndex = 0
  var lifeSqIndex = 0
  var floorIndex = 0
  var areaMIndex = 0

  def parse(data: Array[String]): LabeledPoint = {
    val priceDoc = Math.log(data(priceDocIndex).toDouble+1)

    var fullSq = 40.0

    try {
      fullSq = data(fullSqIndex).toDouble
    }
    catch {
      case e:Exception => println(e)
    }

    val lifeSq = data(lifeSqIndex).toDouble
    val floor = data(floorIndex).toDouble
    val areaM = data(areaMIndex).toDouble

    val label = priceDoc
    val numFeatures = 2
    //var indices = Array(0, 1, 2, 3)
    //var values = Array(fullSq, lifeSq, floor, areaM)
    var indices = Array(0, 1)
    var values = Array(fullSq, lifeSq)

    LabeledPoint(label, Vectors.sparse(numFeatures, indices, values))
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"LinearRegression with $params").setMaster("local")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    val base = "./sberbank/"
    val rawData = sc.textFile(base + "subset.csv")

    val headerAndRows = rawData.map { line => mLine(line) }
    val header = headerAndRows.first

    priceDocIndex = header.indexOf("price_doc")
    fullSqIndex = header.indexOf("full_sq")
    lifeSqIndex = header.indexOf("life_sq")
    floorIndex = header.indexOf("floor")
    areaMIndex = header.indexOf("area_m")

    val data = headerAndRows.filter(_ (0) != header(0))

    val data2 = data.map(parse)

    val splits = data2.randomSplit(Array(0.8, 0.2))
    val training = splits(0).cache()
    val test = splits(1).cache()

    val numTraining = training.count()
    val numTest = test.count()
    println(s"Training: $numTraining, test: $numTest.")

    data.unpersist(blocking = false)

    println(params.regType)
    val updater = params.regType match {
      case NONE => new SimpleUpdater()
      case L1 => new L1Updater()
      case L2 => new SquaredL2Updater()
    }

    training.take(35).foreach(println)

    println("Iterations: " + params.numIterations);
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

  run(defaultParams)
}