import os
import cv2
import numpy as np
from deepface import DeepFace
import logging
from firebase_admin import firestore
import firebase_admin
from config import FIREBASE_CREDENTIALS
from google.cloud.firestore_v1.base_query import FieldFilter
from multiprocessing import Pool
import time

class FaceRecognitionService:
    def __init__(self, dataset_path='/Users/nguyenquangtruong/Desktop/HocTap2/PBL5/SV_SmartClassroom/team_data'):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.dataset_path = dataset_path

        try:
            if not firebase_admin._apps:
                firebase_admin.initialize_app(firebase_admin.credentials.Certificate(FIREBASE_CREDENTIALS))
            self.db = firestore.client()
            self.logger.info("Initialized Firestore client")
        except Exception as e:
            self.logger.error(f"Error initializing Firestore: {str(e)}")
            raise

        self.embedding_cache = {}
        self.current_class_id = None
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            self.logger.info(f"Created dataset directory: {dataset_path}")

    def load_embeddings_for_class(self, class_id):
        if class_id == self.current_class_id and class_id in self.embedding_cache:
            self.logger.info(f"Using cached embeddings for class {class_id}")
            return
        try:
            self.embedding_cache[class_id] = {}
            student_classes = self.db.collection('StudentClasses') \
                .where(filter=FieldFilter('classId', '==', class_id)) \
                .stream()
            student_ids = [sc.to_dict()['studentId'] for sc in student_classes]
            self.logger.info(f"Found {len(student_ids)} students in StudentClasses for class {class_id}")

            for student_id in student_ids:
                student_doc = self.db.collection('Students').document(student_id).get()
                if student_doc.exists:
                    data = student_doc.to_dict()
                    if 'embedding' in data and 'studentName' in data and data['embedding'] and len(data['embedding']) == 128:
                        self.embedding_cache[class_id][data['studentName']] = {
                            'embedding': np.array(data['embedding']),
                            'studentId': data.get('studentId', '')
                        }
                        self.logger.debug(f"Loaded embedding for {data['studentName']} (ID: {student_id})")
                    else:
                        self.logger.warning(f"Student {student_id} missing embedding or studentName or invalid embedding length")
                else:
                    self.logger.warning(f"Student {student_id} not found in Students")

            self.current_class_id = class_id
            self.logger.info(f"Loaded {len(self.embedding_cache[class_id])} embeddings to cache for class {class_id}")
        except Exception as e:
            self.logger.error(f"Error loading embeddings for class {class_id}: {str(e)}")

    def normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def recognize(self, image, class_id):
        try:
            self.load_embeddings_for_class(class_id)

            if not self.embedding_cache.get(class_id):
                self.logger.warning(f"No embeddings loaded for class {class_id}")
                return "Unknown", None

            image = self.resize_image(image)
            self.logger.info("Starting face recognition")
            start_time = time.time()
            embedding = DeepFace.represent(
                image,
                model_name='Facenet',
                detector_backend='retinaface',
                enforce_detection=False
            )
            if not embedding:
                self.logger.warning("No face embedding generated")
                return "Unknown", None
            embedding = self.normalize_embedding(np.array(embedding[0]['embedding']))

            min_distance = float('inf')
            recognized_name = "Unknown"
            recognized_id = None
            threshold = 0.9

            for name, data in self.embedding_cache.get(class_id, {}).items():
                distance = np.linalg.norm(embedding - data['embedding'])
                self.logger.info(f"Distance to {name} (ID: {data['studentId']}): {distance:.2f}")
                if distance < min_distance:
                    min_distance = distance
                    recognized_name = name
                    recognized_id = data['studentId']

            recognize_time = time.time() - start_time
            self.logger.info(f"Completed face recognition, recognized: {recognized_name} (ID: {recognized_id}, distance: {min_distance:.2f}, took {recognize_time:.2f}s)")
            if min_distance < threshold:
                return recognized_name, recognized_id
            else:
                return "Unknown", None
        except Exception as e:
            self.logger.error(f"Error in recognize for class {class_id}: {str(e)}")
            return "Unknown", None

    def add_face(self, name, images, class_id, student_id):
        try:
            embeddings = []
            for image in images:
                image = self.resize_image(image)
                emb = DeepFace.represent(
                    image,
                    model_name='Facenet',
                    detector_backend='retinaface',
                    enforce_detection=False
                )
                if emb:
                    embeddings.append(self.normalize_embedding(np.array(emb[0]['embedding'])))

            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding = self.normalize_embedding(avg_embedding)

                student_ref = self.db.collection('Students').document(student_id)
                student_ref.update({
                    'studentName': name,
                    'embedding': avg_embedding.tolist()
                })

                if class_id not in self.embedding_cache:
                    self.embedding_cache[class_id] = {}
                self.embedding_cache[class_id][name] = {
                    'embedding': avg_embedding,
                    'studentId': student_id
                }
                self.current_class_id = class_id
                self.logger.info(f"Added embedding for {name} (ID: {student_id}) in class {class_id}")
                return True
            else:
                self.logger.error("No valid embeddings generated")
                return False
        except Exception as e:
            self.logger.error(f"Error in add_face for {name} in class {class_id}: {str(e)}")
            return False

    def resize_image(self, img, max_size=640):
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        return img

    @staticmethod
    def process_single_image(args):
        img, img_path, student_name = args
        logger = logging.getLogger(__name__)
        logger.info(f"Processing image: {img_path}")
        try:
            h, w = img.shape[:2]
            max_size = 640
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h))

            emb = DeepFace.represent(
                img,
                model_name='Facenet',
                detector_backend='retinaface',
                enforce_detection=False
            )
            if emb and emb[0]['embedding']:
                embedding = np.array(emb[0]['embedding'])
                norm = np.linalg.norm(embedding)
                embedding = embedding / norm if norm > 0 else embedding
                logger.info(f"Generated embedding for {img_path}")
                return embedding
            else:
                logger.warning(f"No face detected in {img_path}")
                return None
        except Exception as e:
            logger.error(f"Error generating embedding for {img_path}: {str(e)}")
            return None

    def add_student_with_images(self, student_id, student_name, images):
        if len(images) < 7:
            self.logger.error(f"Insufficient images for {student_name}: {len(images)} provided, minimum 7 required")
            return False, "Minimum 7 images required"

        student_folder = os.path.join(self.dataset_path, student_name)
        if not os.path.exists(student_folder):
            os.makedirs(student_folder)
            self.logger.info(f"Created folder for student {student_name}: {student_folder}")

        image_paths = []
        for idx, img in enumerate(images):
            img_path = os.path.join(student_folder, f"{student_id}_{idx}.jpg")
            cv2.imwrite(img_path, img)
            self.logger.info(f"Saved image for {student_name}: {img_path}")
            image_paths.append((img, img_path, student_name))

        try:
            with Pool(processes=os.cpu_count()) as pool:
                embeddings = pool.map(self.process_single_image, image_paths)
        except Exception as e:
            self.logger.error(f"Error in parallel processing: {str(e)}")
            return False, "Parallel processing failed"

        embeddings = [emb for emb in embeddings if emb is not None]
        if not embeddings:
            self.logger.error(f"No valid embeddings generated for {student_name}")
            return False, "Failed to generate embeddings"

        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = self.normalize_embedding(avg_embedding)
        self.logger.info(f"Average embedding shape: {avg_embedding.shape}, values: {avg_embedding[:5]}...")

        try:
            students = self.db.collection('Students').where(filter=FieldFilter('studentId', '==', student_id)).stream()
            student_found = False
            for student in students:
                student_ref = self.db.collection('Students').document(student.id)
                student_ref.update({
                    'studentName': student_name,
                    'embedding': avg_embedding.tolist()
                })
                student_found = True
                self.logger.info(f"Updated embedding for {student_name} (ID: {student_id})")
                break

            if not student_found:
                student_ref = self.db.collection('Students').document(student_id)
                student_data = {
                    'studentId': student_id,
                    'studentName': student_name,
                    'embedding': avg_embedding.tolist(),
                    'email': f"{student_name.lower()}@example.com",
                    'gender': 'Unknown',
                    'phone': '',
                    'studentCode': f"STU_{student_name}",
                    'userId': f"user_{student_name}",
                    'status': 'active'
                }
                self.logger.info(f"Saving student data: {student_data.keys()}")
                student_ref.set(student_data)
                self.logger.info(f"Created new student {student_name} (ID: {student_id})")

            student_doc = self.db.collection('Students').document(student_id).get()
            if student_doc.exists and 'embedding' in student_doc.to_dict():
                self.logger.info(f"Verified: Embedding saved for {student_name} (ID: {student_id}), length: {len(student_doc.to_dict()['embedding'])}")
            else:
                self.logger.error(f"Verification failed: No embedding in Firestore for {student_name} (ID: {student_id})")
                return False, "Failed to save embedding in Firestore"

            return True, "Student added successfully"
        except Exception as e:
            self.logger.error(f"Error saving student {student_name} to Firestore: {str(e)}")
            return False, str(e)