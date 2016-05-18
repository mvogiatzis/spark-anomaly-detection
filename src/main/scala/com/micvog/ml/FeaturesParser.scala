package com.micvog.ml

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD



object FeaturesParser{
  def parseFeatures(rawdata: RDD[String]): RDD[Vector] = {
    val rdd: RDD[Array[Double]] = rawdata.map(_.split(",").map(_.toDouble))
    val vectors: RDD[Vector] = rdd.map(arrDouble => Vectors.dense(arrDouble))
    vectors
  }

  def parseFeaturesWithLabel(cvData: RDD[String]): RDD[LabeledPoint] = {
    val rdd: RDD[Array[Double]] = cvData.map(_.split(",").map(_.toDouble))
    val labeledPoints = rdd.map(arrDouble => new LabeledPoint(arrDouble(0), Vectors.dense(arrDouble.slice(1, arrDouble.length))))
    labeledPoints
  }
}
