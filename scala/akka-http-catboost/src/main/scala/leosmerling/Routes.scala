package leosmerling

import akka.http.scaladsl.server.Directives._
import akka.http.scaladsl.model.StatusCodes
import akka.http.scaladsl.server.Route

import scala.concurrent.Future
import akka.actor.typed.ActorRef
import akka.actor.typed.ActorSystem
import akka.actor.typed.scaladsl.AskPattern._
import akka.util.Timeout

import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport
import io.circe.Decoder.Result
import io.circe.{Decoder, Encoder, HCursor, Json}


trait Protocols extends ErrorAccumulatingCirceSupport {
  import io.circe.generic.semiauto._
  implicit val inputFeaturesDecoder: Decoder[InputFeatures] = deriveDecoder
  implicit val inputFeaturesEncoder: Encoder[InputFeatures] = deriveEncoder
  implicit val predictionResultDecoder: Decoder[PredictionResult] = deriveDecoder
  implicit val predictionResultEncoder: Encoder[PredictionResult] = deriveEncoder
}


//#import-json-formats
//#user-routes-class
class InferenceRoutes(inference: ActorRef[Inference.Command])(implicit val system: ActorSystem[_]) extends Protocols {

  import Inference._

  // If ask takes more time than this to complete the request is failed
  private implicit val timeout: Timeout = Timeout.create(system.settings.config.getDuration("my-app.routes.ask-timeout"))

  def getPrediction(features: InputFeatures): Future[GetPrediction] =
    inference.ask(Predict(features, _))

  val inferenceRoutes: Route =
    pathPrefix("get-prediction") {
      concat(
        //#users-get-delete
        pathEnd {
          concat(
            post {
              entity(as[InputFeatures]) { features =>
                onSuccess(getPrediction(features)) { performed =>
                  complete(performed.result)
                }
              }
            }
          )
        }
      )
    }
}
