package com.incra

import org.apache.commons.io.FileUtils
import org.apache.spark.SparkContext._

object WordCountRunner {

  def main(args: Array[String]) {
    val sc = Helper.getSparkContext(args)

    val lines = sc.textFile("./test-data/wordcount_data.txt", 3)
    val wordCountPairs = lines.flatMap(line => {
      val splitted = line.split(" ")
      splitted.filter(word => word.length > 3).map(word => (word, 1))
    })

    // generate word counts, sort descending by value
    val wordCounts = wordCountPairs.reduceByKey((x: Int, y: Int) => {
      x + y
    }, 10).map(item => item.swap).sortByKey(false)

    // write out partitions as text files
    FileUtils.deleteDirectory(new java.io.File("./results"));
    wordCounts.saveAsTextFile("./results")

    // print out first 10
    wordCounts.take(10).foreach(println(_))

    // block so we can look at http://localhost:4040
    Thread.sleep(1000 * 3600)
  }
}
