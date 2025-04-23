import os
from firebase.FirestoreManager import FirestoreManager
from config import FIREBASE_CREDENTIALS


def get_input(prompt, required=True, type_cast=str):
    while True:
        value = input(prompt).strip()
        if required and not value:
            print("This field is required.")
            continue
        try:
            return type_cast(value) if value else None
        except ValueError as e:
            print(f"Invalid input: {e}")


def check_file_permissions(file_path):
    """Kiểm tra quyền đọc file"""
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    if not os.access(file_path, os.R_OK):
        return f"No read permission for file: {file_path}"
    return None


def main():
    # Debug thông tin môi trường
    print(f"Current working directory: {os.getcwd()}")
    print(f"Firebase credentials path: {FIREBASE_CREDENTIALS}")

    # Kiểm tra file credentials
    error = check_file_permissions(FIREBASE_CREDENTIALS)
    if error:
        print(f"Error: {error}")
        print("Please fix file permissions with: chmod u+r " + FIREBASE_CREDENTIALS)
        return

    try:
        manager = FirestoreManager(FIREBASE_CREDENTIALS)
    except Exception as e:
        print(f"Error initializing FirestoreManager: {e}")
        return

    while True:
        print("\n=== Firestore Admin ===")
        print("1. Alerts")
        print("2. ClassEmotionStats")
        print("3. Classes")
        print("4. StudentClasses")
        print("5. StudentEmotionStats")
        print("6. Students")
        print("7. User")
        print("8. Exit")

        choice = get_input("Select collection (1-8): ", type_cast=int)
        if choice == 8:
            print("Exiting...")
            break
        if choice not in range(1, 8):
            print("Invalid choice.")
            continue

        collection_map = {
            1: "Alerts",
            2: "ClassEmotionStats",
            3: "Classes",
            4: "StudentClasses",
            5: "StudentEmotionStats",
            6: "Students",
            7: "User"
        }
        collection = collection_map[choice]

        print(f"\n=== {collection} ===")
        print("1. Add document")
        print("2. Delete document")
        action = get_input("Select action (1-2): ", type_cast=int)
        if action not in [1, 2]:
            print("Invalid action.")
            continue

        try:
            if action == 2:
                doc_id = get_input("Enter documentId to delete: ")
                # Xử lý riêng cho Alerts
                method_name = "delete_alert" if collection == "Alerts" else f"delete_{collection.lower()}"
                getattr(manager, method_name)(doc_id)
                continue

            # Thêm document
            if collection == "Alerts":
                class_id = get_input("Class ID: ")
                content = get_input("Content: ")
                timestamp = get_input("Timestamp (YYYY-MM-DD HH:MM:SS or 'now'): ")
                title = get_input("Title: ")
                manager.add_alert(class_id, content, timestamp, title)

            elif collection == "ClassEmotionStats":
                class_id = get_input("Class ID: ")
                angry = get_input("Angry (%): ", type_cast=float)
                fear = get_input("Fear (%): ", type_cast=float)
                happy = get_input("Happy (%): ", type_cast=float)
                neutral = get_input("Neutral (%): ", type_cast=float)
                sad = get_input("Sad (%): ", type_cast=float)
                surprise = get_input("Surprise (%): ", type_cast=float)
                total_detected = get_input("Total Detected Students: ", type_cast=int)
                create_at = get_input("Create At (YYYY-MM-DD HH:MM:SS or 'now'): ")
                end_time = get_input("End Time (YYYY-MM-DD HH:MM:SS or 'now'): ")
                start_time = get_input("Start Time (YYYY-MM-DD HH:MM:SS or 'now'): ")
                manager.add_class_emotion_stats(class_id, angry, fear, happy, neutral, sad, surprise, total_detected,
                                                create_at, end_time, start_time)

            elif collection == "Classes":
                class_id = get_input("Class ID: ")
                class_name = get_input("Class Name: ")
                day_of_week = get_input("Day of Week: ")
                description = get_input("Description: ", required=False)
                end_time = get_input("End Time (YYYY-MM-DD HH:MM:SS or 'now'): ")
                start_time = get_input("Start Time (YYYY-MM-DD HH:MM:SS or 'now'): ")
                user_id = get_input("User ID: ")
                manager.add_class(class_id, class_name, day_of_week, description, end_time, start_time, user_id)

            elif collection == "StudentClasses":
                class_id = get_input("Class ID: ")
                student_id = get_input("Student ID: ")
                joined_at = get_input("Joined At (YYYY-MM-DD HH:MM:SS): ")
                manager.add_student_class(class_id, student_id, joined_at)

            elif collection == "StudentEmotionStats":
                class_id = get_input("Class ID: ")
                angry = get_input("Angry (%): ", type_cast=float)
                fear = get_input("Fear (%): ", type_cast=float)
                happy = get_input("Happy (%): ", type_cast=float)
                neutral = get_input("Neutral (%): ", type_cast=float)
                sad = get_input("Sad (%): ", type_cast=float)
                surprise = get_input("Surprise (%): ", type_cast=float)
                total_detection = get_input("Total Detection: ", type_cast=int)
                create_at = get_input("Create At (YYYY-MM-DD HH:MM:SS or 'now'): ")
                end_time = get_input("End Time (YYYY-MM-DD HH:MM:SS or 'now'): ")
                start_time = get_input("Start Time (YYYY-MM-DD HH:MM:SS or 'now'): ")
                student_id = get_input("Student ID: ")
                manager.add_student_emotion_stats(class_id, angry, fear, happy, neutral, sad, surprise, total_detection,
                                                  create_at, end_time, start_time, student_id)

            elif collection == "Students":
                avatar_url = get_input("Avatar URL: ", required=False)
                date_of_birth = get_input("Date of Birth (YYYY-MM-DD): ")
                email = get_input("Email: ")
                gender = get_input("Gender: ")
                notes = get_input("Notes: ", required=False)
                phone = get_input("Phone: ", required=False)
                status = get_input("Status: ")
                student_code = get_input("Student Code: ")
                student_name = get_input("Student Name: ")
                user_id = get_input("User ID: ")
                manager.add_student(avatar_url, date_of_birth, email, gender, notes, phone, status, student_code,
                                    student_name, user_id)

            elif collection == "User":
                email = get_input("Email: ")
                role = get_input("Role: ")
                username = get_input("Username: ")
                manager.add_user(email, role, username)

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()