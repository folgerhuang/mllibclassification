import breeze.linalg._
import org.apache.spark.mllib.feature.{IDFModel, HashingTF, IDF}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{SparseVector => SV}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source

/**
  * Created by Administrator on 2016/10/25 0025.
  */
object Tfidf {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("TfIdfTest").setMaster("local")
    val sc = new SparkContext(conf)
//    val a: RDD[Int] = sc.parallelize(Array(1,2,3))
//    val b: RDD[Int] = sc.parallelize(Array(4,5,6))
//    val c: RDD[(Int, Int)] = a.zip(b)
//    c.foreach(println(_))
//    �ļ�����ÿ�д���һ��document������zipһ��index��Ϊ�˱�ʾ����
    val documents: RDD[(Seq[String], Long)] = sc.parallelize(Source.fromFile("E:\\�γ�\\mllib\\mllib��һ��\\�γ̸�������\\doc.txt").getLines().filter(_.trim.length > 0).toSeq).map(_.split(" ").toSeq).zipWithIndex()

//    ����TF���󣬲�������ϡ��������ĳ��ȣ����ﲻдĬ��1<<20
    val hashingTF = new HashingTF(Math.pow(2, 18).toInt)
    val tf_num_pairs: RDD[(Long, linalg.Vector)] = documents.map {
      case (seq, num) =>
//        ����ÿ���ĵ��Ĵ�Ƶ
        val tf: linalg.Vector = hashingTF.transform(seq)
        (num, tf)
    }

    tf_num_pairs.cache()
//    �������еĴ�Ƶ�������Ƶ
    val idf: IDFModel = new IDF().fit(tf_num_pairs.values)
    val num_idf_pairs: RDD[(Long, linalg.Vector)] = tf_num_pairs.mapValues(v => idf.transform(v))
//    ����������֮������ƶȣ���Ҫ�ѿ����������Լ��㲥һ��
    val b_num_idf_pairs = sc.broadcast(num_idf_pairs.collect())



//��document�����ƶ�
//    val docSims = num_idf_pairs.flatMap {
//      case (id1, idf1) =>
//        val idfs: Array[(Long, linalg.Vector)] = b_num_idf_pairs.value.filter(_._1 != id1)
//        val sv1: SV = idf1.asInstanceOf[SV]
//        val bsv1 = new SparseVector[Double](sv1.indices, sv1.values, sv1.size)
//        idfs.map {
//          case (id2, idf2) =>
//            val sv2 = idf2.asInstanceOf[SV]
//            val bsv2 = new SparseVector[Double](sv2.indices, sv2.values, sv2.size)
//            val cosSim = bsv1.dot(bsv2).asInstanceOf[Double] / (norm(bsv1) * norm(bsv2))
//            (id1, id2, cosSim)
//        }

        val docSims = num_idf_pairs.flatMap {
          case (id1, idf1) =>
//            �����Լ����Լ����ƶȣ����Թ��˺��Լ�idһ����
            val idfs: Array[(Long, linalg.Vector)] = b_num_idf_pairs.value.filter(_._1 != id1)
            val sv1: SV = idf1.asInstanceOf[SV]
           val data1 = sv1.toArray
//            �������зǵ�ǰ�ĵ����Ƚ�����֮������ƶ�
            idfs.map {
              case (id2, idf2) =>
                val sv2 = idf2.asInstanceOf[SV]
                val data2 = sv2.toArray
//                ��������ÿ��Ԫ�����֮��
                val AB = data1.zip(data2).map(sa=>{sa._1*sa._2}).reduce(_+_)
//                ��������ÿ��Ԫ��ƽ��֮���ٿ���
                val squartA = Math.sqrt(data1.map(sa=>{Math.pow(sa,2)}).reduce(_+_))
                val squartB = Math.sqrt(data2.map(sa=>{Math.pow(sa,2)}).reduce(_+_))
//                ���빫ʽ
                val cosSim = AB/(squartA*squartB)
                (id1, id2, cosSim)
            }
    }






    docSims.foreach(println(_))

    sc.stop()

  }
}
