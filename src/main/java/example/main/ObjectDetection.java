package example.main;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ObjectDetection {

	private static final Logger logger = LoggerFactory.getLogger(ObjectDetection.class);

	private ObjectDetection() {
	}

	public static void main(String[] args) throws IOException, ModelException, TranslateException {
		DetectedObjects detection = ObjectDetection.predict();
		logger.info("{}", detection);
	}

	public static DetectedObjects predict() throws IOException, ModelException, TranslateException {
		Path imageFile = Paths.get("src/test/resources/human-5563795_1280.jpg");
		Image img = ImageFactory.getInstance().fromFile(imageFile);
		Criteria<Image, DetectedObjects> criteria = Criteria.builder().optApplication(Application.CV.OBJECT_DETECTION)
				.setTypes(Image.class, DetectedObjects.class).optFilter("backbone", "resnet50")
				.optProgress(new ProgressBar()).build();
		try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
			try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
				DetectedObjects detection = predictor.predict(img);
				saveBoundingBoxImage(img, detection);
				return detection;
			}
		}
	}

	private static void saveBoundingBoxImage(Image img, DetectedObjects detection) throws IOException {
		Path outputDir = Paths.get("build/output");
		Files.createDirectories(outputDir);
		Image newImage = img.duplicate(Image.Type.TYPE_INT_ARGB);
		newImage.drawBoundingBoxes(detection);
		Path imagePath = outputDir.resolve("detected-objects.png");
		newImage.save(Files.newOutputStream(imagePath), "png");
		logger.info("Detected objects image has been saved in: {}", imagePath);
	}
}