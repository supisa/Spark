import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

val conf = new SparkConf().setAppName("Forest")
val sc = new SparkContext(conf)

val rawData = sc.textFile("C:/Users/OWNER/data/covtype.data")
val preData = rawData.map{line =>
  val values = line.split(',').map(_.toDouble)
  Vectors.dense(values)
}
val summary: MultivariateStatisticalSummary = Statistics.colStats(preData)

val data = rawData.map{line =>
  val values = line.split(',').map(_.toDouble)
  val featureVector = Vectors.dense(values.init)
  val label = values.last - 1
  LabeledPoint(label,featureVector)
 }

val Array(trainData, cvData, testData) =
  data.randomSplit(Array(0.8,0.1,0.1))
trainData.cache()
cvData.cache()
testData.cache()

def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]):
    MulticlassMetrics = {
  val predictionAndLabels = data.map(example =>
    (model.predict(example.features),example.label)
  )
  new MulticlassMetrics(predictionAndLabels )
}

val model = DecisionTree.trainClassifier(
  trainData, 7, Map[Int,Int](), "gini", 4, 100)

val metrics = getMetrics(model,cvData)
//metrics.confusionMatrix
//metrics.precision

(0 until 7).map(
  cat => (metrics.precision(cat),metrics.recall(cat))
).foreach(println)

def classProbabilities(data: RDD[LabeledPoint]):Array[Double] = {
  val countsByCategory = data.map(_.label).countByValue()
  val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
  counts.map(_.toDouble/counts.sum)
}

val trainPriorProbabilities = classProbabilities(trainData)
val cvPriorProbabilities = classProbabilities(cvData)
trainPriorProbabilities.zip(cvPriorProbabilities).map {
  case (trainProb, cvProb) => cvProb
}.sum

//the achieved accuracy of random guessing

val evaluations =
  for ( impurity  <- Array("gini","entropy");
        depth     <- Array(1,20);
        bins      <- Array(10,300))
    yield {
      val model =DecisionTree.trainClassifier(trainData, 7, Map[Int,Int](), impurity, depth, bins)
      val predictionsAndLabels = cvData.map(example =>
        (model.predict(example.features),example.label)
      )
      val accuracy =
        new MulticlassMetrics(predictionsAndLabels).precision
      ((impurity, depth, bins),accuracy)
    }
evaluations.sortBy(_._2).reverse.foreach(println)
val forest = RandomForest.trainClassifier(
  trainData, 7, Map[Int,Int](), 20, "auto", "entropy",30 ,300)

//Making Prediction
val input = "2709, 125, 28, 23, 3224, 253, 207, 61, 6094,0 ,29"
val vector = Vectors.dense(input.split('.').map(_.toDouble))
forest.predict(vector)


