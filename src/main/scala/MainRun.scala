/* SimpleApp.scala */
import com.micvog.ml.AnomalyDetection
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD


object MainRun {



  def main(args: Array[String]) {
    val rawFilePath = "/Users/micvog/tools/spark/data/features.csv" // Should be some file on your system
    val cvFilePath = "/Users/micvog/tools/spark/data/cv.csv"

    val conf = new SparkConf().setAppName("Anomaly Detection Spark")
    val sc = new SparkContext(conf)

    val rawdata = sc.textFile(rawFilePath, 2).cache()
    val cvData = sc.textFile(cvFilePath, 2).cache()

    //convert raw data to vectors
    val vectors: RDD[Vector] = parseFeatures(rawdata)
    val cvVectors: RDD[LabeledPoint] = parseFeaturesWithLabel(cvData)

    val data = vectors.cache()
    val anDet: AnomalyDetection = new AnomalyDetection()
    //derive model
    val model = anDet.run(data)

    val dataCvVec = cvVectors.cache()
    val optimalModel = anDet.optimize(dataCvVec, model)
  }

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