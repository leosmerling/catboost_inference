package leosmerling

//#user-registry-actor
import akka.actor.typed.ActorRef
import akka.actor.typed.Behavior
import akka.actor.typed.scaladsl.Behaviors
import scala.collection.immutable
import scala.language.postfixOps

import ai.catboost.CatBoostModel
import ai.catboost.CatBoostPredictions

import cb.cb.Features
import cb.cb.Prediction

// case class PredictionResult(
//     predicted_days: Float
// )

// case class InputFeatures(
//     f1: Float,
//     f2: Float,
//     catf1: String, 
//     catf2: String, 
//     catf3: String, 
// )

object Inference {
    //actor protocol
    sealed trait Command

    final case class Predict(features: Array[Features], replyTo: ActorRef[GetPrediction]) extends Command

    final case class GetPrediction(result: Seq[Prediction], latency: Long)

    def loadModel(): CatBoostModel = {
        println("Loading model...")
        CatBoostModel.loadModel("src/main/resources/model.cbm")
    }

    def apply() = predict(loadModel())

    val mxBean = java.lang.management.ManagementFactory.getPlatformMXBean(classOf[java.lang.management.ThreadMXBean])

    def predict(model: CatBoostModel): Behavior[Command] = 
        Behaviors.receiveMessage {
            case Predict(features, replyTo) =>
                val t0 = mxBean.getCurrentThreadCpuTime()
                val predictions = model.predict(
                    features.map(x => Array[Float](x.floatFeature4, x.floatFeature5)),
                    features.map(x => Array[String](x.catFeature1, x.catFeature2, x.catFeature3)),
                )
                val t1 = mxBean.getCurrentThreadCpuTime()
                replyTo ! GetPrediction(
                    (0 until features.length).map(i => Prediction(predictions.get(i, 0).toFloat)),
                    (t1 - t0).toLong
                )
                Behaviors.same
        }
}
