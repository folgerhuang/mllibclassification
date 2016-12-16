import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{GradientBoostedTrees, DecisionTree}
import org.apache.spark.mllib.tree.configuration.{BoostingStrategy, Algo}
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/8/21 0021.
  */
object GBDT_new {
//  E:\课程\推荐系统\资料\train\train F:\额外项目\pensionRisk\data\DecisionTree\model 10 local
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    if (args.length < 4) {
      System.err.println("Usage: DecisionTrees <inputPath> <modelPath> <maxDepth> <master> [<AppName>]")
      System.err.println("eg: hdfs://192.168.57.104:8020/user/000000_0 10 0.1 spark://192.168.57.104:7077 DecisionTrees")
      System.exit(1)
    }
    val appName = if(args.length>4) args(4) else "DecisionTrees"
    val conf = new SparkConf().setAppName(appName).setMaster(args(3))
    val sc = new SparkContext(conf)
    val traindata: RDD[LabeledPoint] = MLUtils.loadLabeledPoints(sc,args(0))
    val features = traindata.map(_.features)
    val scaler: StandardScalerModel = new StandardScaler(withMean=true,withStd=true).fit(features)
    val train: RDD[LabeledPoint] = traindata.map(sample=>{
      val label = sample.label
      val feature = scaler.transform(sample.features)
      new LabeledPoint(label,feature)
    })
  val splitRdd: Array[RDD[LabeledPoint]] = traindata.randomSplit(Array(1.0,9.0))
  val testData: RDD[LabeledPoint] = splitRdd(0)
  val realTrainData = splitRdd(1)

  val boostingStrategy = BoostingStrategy.defaultParams("Classification")
  boostingStrategy.setNumIterations(3)
  boostingStrategy.treeStrategy.setNumClasses(2)
  boostingStrategy.treeStrategy.setMaxDepth(args(2).toInt)
  boostingStrategy.setLearningRate(0.8)
  boostingStrategy.treeStrategy.setCategoricalFeaturesInfo(Map[Int, Int]())
  val model = GradientBoostedTrees.train(realTrainData, boostingStrategy)

  val labelAndPreds = testData.map(point =>{
    val prediction = model.predict(point.features)
    (point.label, prediction)
  })
  val acc = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
  println("Test Error = " + acc)
  println("Learned classification GBT model:\n" + model.toDebugString)

  model.save(sc, "")
  }
}
