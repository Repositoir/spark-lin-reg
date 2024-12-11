package nl.aritra.dataeng

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

object Main {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("spark-scala")
      .master("local[*]")
      .config("spark.driver.bindAddress", "127.0.0.1")
      // .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
      .getOrCreate()

    val testData = spark.read
      .option("header", value = true)
      .option("inferSchema", value = true)
      .csv("data/test.csv")

    val testDataFilled = testData.na.fill(Map("Age" -> 30.0, "Fare" -> 0.0, "Cabin" -> "None", "Embarked" -> "S"))

    val trainData = spark.read
      .option("header", value = true)
      .option("inferSchema", value = true)
      .csv("data/train.csv")

    val trainDataFilled = trainData.na.fill(Map("Age" -> 30.0, "Fare" -> 0.0, "Cabin" -> "None", "Embarked" -> "S"))

    val sexIndexer = new StringIndexer()
      .setInputCol("Sex")
      .setOutputCol("SexIndexed")

    val embarkedIndexer = new StringIndexer()
      .setInputCol("Embarked")
      .setOutputCol("EmbarkedIndexed")

    val indexedTrainData = new Pipeline()
      .setStages(Array(sexIndexer, embarkedIndexer))
      .fit(trainDataFilled)
      .transform(trainDataFilled)

    val featureColumns = Array("SexIndexed", "EmbarkedIndexed", "Age", "Fare")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val assembledData = vectorAssembler.transform(indexedTrainData)
    // assembledData.select("features").show(false)

    val dataWithLabel = assembledData.withColumnRenamed("Survived", "label")

    val Array(trainingData, testingData) = dataWithLabel.randomSplit(Array(0.8, 0.2), seed = 1234)

    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxIter(100)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    val lrModel = lr.fit(trainingData)

    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    val predictions = lrModel.transform(testingData)
    predictions.select("label", "features", "prediction").show(false)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error: $rmse")

  }
}