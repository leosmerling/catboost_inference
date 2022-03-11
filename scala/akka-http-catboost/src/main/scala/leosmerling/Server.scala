package leosmerling

import akka.actor.typed.ActorSystem
import akka.actor.typed.scaladsl.Behaviors
import akka.http.scaladsl.Http
import akka.http.scaladsl.server.Route

import scala.util.Failure
import scala.util.Success


//#main-class
object QuickstartApp {
  //#start-http-server
  private def startHttpServer(routes: Route)(implicit system: ActorSystem[_]): Unit = {
    // Akka HTTP still needs a classic ActorSystem to start
    import system.executionContext

    val futureBinding = Http().newServerAt("localhost", 8080).bind(routes)
    futureBinding.onComplete {
      case Success(binding) =>
        val address = binding.localAddress
        system.log.info("Server online at http://{}:{}/", address.getHostString, address.getPort)
      case Failure(ex) =>
        system.log.error("Failed to bind HTTP endpoint, terminating system", ex)
        system.terminate()
    }
  }
  //#start-http-server
  def main(args: Array[String]): Unit = {
    //#server-bootstrapping
    val rootBehavior = Behaviors.setup[Nothing] { context =>
      val inferenceActor = context.spawn(Inference(), "InferenceActor")
      context.watch(inferenceActor)

      val routes = new InferenceRoutes(inferenceActor)(context.system)
      startHttpServer(routes.inferenceRoutes)(context.system)

      Behaviors.empty
    }
    val system = ActorSystem[Nothing](rootBehavior, "InferenceAkkaHttpServer")
    //#server-bootstrapping
  }
}
//#main-class

// import akka.actor.ActorSystem
// import akka.event.{Logging, LoggingAdapter}
// import akka.http.scaladsl.Http
// import akka.http.scaladsl.client.RequestBuilding
// import akka.http.scaladsl.marshalling.ToResponseMarshallable
// import akka.http.scaladsl.model.{HttpRequest, HttpResponse}
// import akka.http.scaladsl.model.StatusCodes._
// import akka.http.scaladsl.server.Directives._
// import akka.http.scaladsl.server.Route
// import akka.http.scaladsl.unmarshalling.Unmarshal
// import akka.stream.scaladsl.{Flow, Sink, Source}
// import com.typesafe.config.Config
// import com.typesafe.config.ConfigFactory
// import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport
// import io.circe.Decoder.Result
// import io.circe.{Decoder, Encoder, HCursor, Json}

// import scala.concurrent.{ExecutionContext, Future}
// import scala.math._


// trait Protocols extends ErrorAccumulatingCirceSupport {
//   import io.circe.generic.semiauto._
//   implicit val predictionRequestDecoder: Decoder[InputFeatures] = deriveDecoder
//   implicit val predictionResponseDecoder: Decoder[PredictionResult] = deriveDecoder
// }

// trait Service extends Protocols {
//   implicit val system: ActorSystem
//   implicit def executor: ExecutionContext

//   def config: Config
//   val logger: LoggingAdapter

//   val routes: Route = {
//     logRequestResult("akka-http-catboost") {
//       pathPrefix("health") {
//         (get & path(Segment)) { _ =>
//           complete { Future("OK").map[ToResponseMarshallable] }
//         }
//       } ~
//       pathPrefix("predict") {
//         (post & entity(as[PredictionRequest])) { req =>
//           complete {
//             Future(PredictionResponse(1.0f)).map[ToResponseMarshallable]
//           }
//         }
//       }
//     }
//   }
// }

// object AkkaHttpMicroservice extends App with Service {
//   override implicit val system: ActorSystem = ActorSystem()
//   override implicit val executor: ExecutionContext = system.dispatcher

//   override val config = ConfigFactory.load()
//   override val logger = Logging(system, "AkkaHttpMicroservice")

//   Http()
//     .newServerAt(config.getString("http.interface"), config.getInt("http.port"))
//     .bindFlow(routes)
// }
