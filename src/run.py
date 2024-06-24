from time import time

import cv2
import face_recognition as fr

from embeddings_manager import EmbeddingsManager

MATCH_COLOR = (0, 255, 0)
UNKNOWN_COLOR = (255, 0, 0)


if __name__ == "__main__":
    vid = cv2.VideoCapture(0)
    embedding_manager = EmbeddingsManager()

    while True:
        start = time()
        ret, frame = vid.read()

        faces_locations = fr.face_locations(frame)
        embeddings = fr.face_encodings(frame, known_face_locations=faces_locations)
        for loc, emb in zip(faces_locations, embeddings):
            match, name, distance = embedding_manager.check_embedding(emb)
            color = MATCH_COLOR if match else UNKNOWN_COLOR

            frame = cv2.rectangle(frame, (loc[1], loc[0]), (loc[3], loc[2]), color, 3)
            cv2.putText(
                frame,
                name + " " + str(round(distance, 2)),
                (loc[3], loc[2] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
                cv2.LINE_AA,
            )

        t = time() - start
        FPS = round(1 / t, 1)
        cv2.putText(frame, str(FPS), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        # frame = сv2.resize(frame, (1000,1000))
        frame = cv2.resize(frame, (1200, 900), interpolation=cv2.INTER_AREA)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()
