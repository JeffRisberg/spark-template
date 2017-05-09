package com.incra.dataframes

import org.apache.spark._

case class Person(name: String, age: Int, salary: Int)

object Example01 {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Example01").setMaster("local")
    val sc = new SparkContext(conf)

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    import sqlContext.implicits._

    // Create an RDD of Person objects and register it as a table.
    val textRows = sc.textFile("./test-data/people.txt")
    val peopleRDD = textRows.map(_.split(",")).map(p => Person(p(0), p(1).trim.toInt, p(2).trim.toInt))
    val people = peopleRDD.toDF()
    people.registerTempTable("people")

    println(people.count())
    people.columns foreach println
    people.show()

    people.filter(people("age") > 21).show()

    people.groupBy().agg(Map("age" -> "max", "salary" -> "avg")).show()

    println(people.rdd.count)
    people.rdd foreach println
  }
}
