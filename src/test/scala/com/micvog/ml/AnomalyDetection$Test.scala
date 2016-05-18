package com.micvog.ml

import com.holdenkarau.spark.testing.SharedSparkContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.scalactic.Equality
import org.scalatest.{FlatSpec, FunSuite, Matchers}

class AnomalyDetection$Test extends FlatSpec with Matchers with SharedSparkContext {

  val point = Vectors.dense(Array(14.8593411857427, 14.9006647394062))
  val means = Vectors.dense(Array(14.1122257839456, 14.9977105081362))
  val variances = Vectors.dense(Array(1.83263141349452, 1.70974533082878))

  "probFunction" should "return correct product value" in {
    val p = AnomalyDetection.probFunction(point, means, variances)
    assert(p === 0.0769984879544 +- 0.0001)
  }

  "predict" should "predict the anomaly" in {
    assert(!AnomalyDetection.predict(point, means, variances, 0.05))
  }

  "predict" should "predict non anomaly" in {
    assert(AnomalyDetection.predict(point, means, variances, 0.08))
  }

  private def vectorequality() = {
    new Equality[Vector] {
      def areEqual(a: Vector, b: Any): Boolean =
        b match {
          case v: Vector => v.toArray.zip(a.toArray).map(pair => pair._1 === pair._2 +- 0.001).reduce((a, b) => a && b)
          case _ => false
        }
    }
  }

  def trainModel(): AnomalyDetectionModel = {
    val trainingExamplesFilePath = "./src/test/resources/training.csv"
    val trainingData = sc.textFile(trainingExamplesFilePath, 2).cache()
    val trainingRdd = FeaturesParser.parseFeatures(trainingData)
    new AnomalyDetection().run(trainingRdd)
  }

  "run" should "return model with correct mean and variance" in {
    val model: AnomalyDetectionModel = trainModel()

    //use scalactic's more relaxing equality
    implicit val vectorEq = vectorequality()

    assert(model.means === Vectors.dense(Array(79.9843751617201, 5.13662727300755)))
    assert(model.variances === Vectors.dense(Array(356.44539323536225, 3.79818173645375)))
  }

  "optimize" should "calculate epsilon and F1 score" in {
    val cvFilePath = "./src/test/resources/cross_val.csv"
    val cvData = sc.textFile(cvFilePath, 2).cache()
    val cvPointsRdd: RDD[LabeledPoint] = FeaturesParser.parseFeaturesWithLabel(cvData)

    val model = trainModel()
    val optimalModel = new AnomalyDetection().optimize(cvPointsRdd, model)
    assert(optimalModel.epsilon === 3.382218E-4 +- 0.0000000001)
  }

}
