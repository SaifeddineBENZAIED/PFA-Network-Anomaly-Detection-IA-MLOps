from django.shortcuts import render
from django.http import QueryDict
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework import serializers, views, status
from rest_framework.response import Response
import pandas as pd
import tensorflow as tf
import numpy as np
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Chargez le modèle une fois au démarrage de l'application
model_path = 'C:/Users/benzaied saif/Desktop/PFA_web/BackEnd_django/pfa_web_backend/pfaBackend_app/myModels/cnn_full_model.h5'
try:
    model = tf.keras.models.load_model(model_path)
except OSError as e:
    print(f"Error loading model: {e}")

# Define the class names mapping
LABELS_MAPPING = {
    0: 'BENIGN',
    1: 'Bot',
    2: 'DDoS',
    3: 'DoS GoldenEye',
    4: 'DoS Hulk',
    5: 'DoS Slowhttptest',
    6: 'DoS slowloris',
    7: 'FTP-Patator',
    8: 'Heartbleed',
    9: 'Infiltration',
    10: 'PortScan',
    11: 'SSH-Patator',
    12: 'Web Attack-Brute Force',
    13: 'Web Attack-Sql Injection',
    14: 'Web Attack-XSS'
}

class DataInputSerializer(serializers.Serializer):
    Destination_Port = serializers.IntegerField(default=0)
    Flow_Duration = serializers.FloatField(default=0.0)
    Total_Fwd_Packets = serializers.IntegerField(default=0)
    Total_Backward_Packets = serializers.IntegerField(default=0)
    Total_Length_of_Fwd_Packets = serializers.FloatField(default=0.0)
    Total_Length_of_Bwd_Packets = serializers.FloatField(default=0.0)
    Fwd_Packet_Length_Max = serializers.FloatField(default=0.0)
    Fwd_Packet_Length_Min = serializers.FloatField(default=0.0)
    Fwd_Packet_Length_Mean = serializers.FloatField(default=0.0)
    Fwd_Packet_Length_Std = serializers.FloatField(default=0.0)
    Bwd_Packet_Length_Max = serializers.FloatField(default=0.0)
    Bwd_Packet_Length_Min = serializers.FloatField(default=0.0)
    Bwd_Packet_Length_Mean = serializers.FloatField(default=0.0)
    Bwd_Packet_Length_Std = serializers.FloatField(default=0.0)
    Flow_Bytes_s = serializers.FloatField(default=0.0)
    Flow_Packets_s = serializers.FloatField(default=0.0)
    Flow_IAT_Mean = serializers.FloatField(default=0.0)
    Flow_IAT_Std = serializers.FloatField(default=0.0)
    Flow_IAT_Max = serializers.FloatField(default=0.0)
    Flow_IAT_Min = serializers.FloatField(default=0.0)
    Fwd_IAT_Total = serializers.FloatField(default=0.0)
    Fwd_IAT_Mean = serializers.FloatField(default=0.0)
    Fwd_IAT_Std = serializers.FloatField(default=0.0)
    Fwd_IAT_Max = serializers.FloatField(default=0.0)
    Fwd_IAT_Min = serializers.FloatField(default=0.0)
    Bwd_IAT_Total = serializers.FloatField(default=0.0)
    Bwd_IAT_Mean = serializers.FloatField(default=0.0)
    Bwd_IAT_Std = serializers.FloatField(default=0.0)
    Bwd_IAT_Max = serializers.FloatField(default=0.0)
    Bwd_IAT_Min = serializers.FloatField(default=0.0)
    Fwd_PSH_Flags = serializers.IntegerField(default=0)
    Bwd_PSH_Flags = serializers.IntegerField(default=0)
    Fwd_URG_Flags = serializers.IntegerField(default=0)
    Bwd_URG_Flags = serializers.IntegerField(default=0)
    Fwd_Header_Length = serializers.IntegerField(default=0)
    Bwd_Header_Length = serializers.IntegerField(default=0)
    Fwd_Packets_s = serializers.FloatField(default=0.0)
    Bwd_Packets_s = serializers.FloatField(default=0.0)
    Min_Packet_Length = serializers.FloatField(default=0.0)
    Max_Packet_Length = serializers.FloatField(default=0.0)
    Packet_Length_Mean = serializers.FloatField(default=0.0)
    Packet_Length_Std = serializers.FloatField(default=0.0)
    Packet_Length_Variance = serializers.FloatField(default=0.0)
    FIN_Flag_Count = serializers.IntegerField(default=0)
    SYN_Flag_Count = serializers.IntegerField(default=0)
    RST_Flag_Count = serializers.IntegerField(default=0)
    PSH_Flag_Count = serializers.IntegerField(default=0)
    ACK_Flag_Count = serializers.IntegerField(default=0)
    URG_Flag_Count = serializers.IntegerField(default=0)
    CWE_Flag_Count = serializers.IntegerField(default=0)
    ECE_Flag_Count = serializers.IntegerField(default=0)
    Down_Up_Ratio = serializers.FloatField(default=0.0)
    Average_Packet_Size = serializers.FloatField(default=0.0)
    Avg_Fwd_Segment_Size = serializers.FloatField(default=0.0)
    Avg_Bwd_Segment_Size = serializers.FloatField(default=0.0)
    Fwd_Header_Length_1 = serializers.IntegerField(default=0)
    Fwd_Avg_Bytes_Bulk = serializers.FloatField(default=0.0)
    Fwd_Avg_Packets_Bulk = serializers.FloatField(default=0.0)
    Fwd_Avg_Bulk_Rate = serializers.FloatField(default=0.0)
    Bwd_Avg_Bytes_Bulk = serializers.FloatField(default=0.0)
    Bwd_Avg_Packets_Bulk = serializers.FloatField(default=0.0)
    Bwd_Avg_Bulk_Rate = serializers.FloatField(default=0.0)
    Subflow_Fwd_Packets = serializers.IntegerField(default=0)
    Subflow_Fwd_Bytes = serializers.IntegerField(default=0)
    Subflow_Bwd_Packets = serializers.IntegerField(default=0)
    Subflow_Bwd_Bytes = serializers.IntegerField(default=0)
    Init_Win_bytes_forward = serializers.IntegerField(default=0)
    Init_Win_bytes_backward = serializers.IntegerField(default=0)
    act_data_pkt_fwd = serializers.IntegerField(default=0)
    min_seg_size_forward = serializers.FloatField(default=32.0)
    Active_Mean = serializers.FloatField(default=0.0)
    Active_Std = serializers.FloatField(default=0.0)
    Active_Max = serializers.FloatField(default=0.0)
    Active_Min = serializers.FloatField(default=0.0)
    Idle_Mean = serializers.FloatField(default=0.0)
    Idle_Std = serializers.FloatField(default=0.0)
    Idle_Max = serializers.FloatField(default=0.0)
    Idle_Min = serializers.FloatField(default=0.0)


def preprocess_single_data(data: dict) -> np.ndarray:
    # Create DataFrame from input data
    df = pd.DataFrame([data])
    print("Initial data:\n", df)
    """
    # Replace infinite values and set negative values to NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[df < 0] = np.nan
    print("Data after inf handling:\n", df)

    # Fill NaN values with forward fill and backfill
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    print("Data after NaN handling:\n", df)

    # Check if there are still any NaNs left
    if df.isna().any().any():
        df.fillna(0, inplace=True)  # Filling remaining NaNs with 0 might be necessary if ffill/bfill doesn't work
    """
    # Reshape data to required input shape for model
    reshaped_data = reshape_single_data_for_cnn(df)
    print("Reshaped data:\n", reshaped_data)

    return reshaped_data

def reshape_single_data_for_cnn(x: np.ndarray) -> np.ndarray:
    required_padding = 81 - x.shape[1]
    padded_x = np.pad(x, ((0, 0), (0, required_padding)), 'constant', constant_values=0)
    reshaped_x = np.reshape(padded_x, (1, 9, 9, 1))  # Reshape to (batch_size, height, width, channels)
    return reshaped_x

class AnalyzeCsvView(views.APIView):
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request, *args, **kwargs):
        def interpret_predictions(predictions):
            sorted_indices = np.argsort(predictions[0])[::-1]
            sorted_classes = [
                f"{LABELS_MAPPING.get(index)}: {predictions[0][index] * 100:.2f}%"
                for index in sorted_indices
            ]
            return sorted_classes

        try:
            if 'file' in request.FILES:
                file = request.FILES['file']
                df = pd.read_csv(file)
                data_dict = df.iloc[0].to_dict()

            elif request.content_type == 'application/json':
                data_dict = request.data

            else:
                return Response({"error": "Invalid request format"}, status=status.HTTP_400_BAD_REQUEST)

            logger.info(f"Data input: {data_dict}")
            print("Data input: ", data_dict)

            processed_data = preprocess_single_data(data_dict)
            logger.info(f"Processed data: {processed_data}")
            print("Processed data: ", processed_data)

            predictions = model.predict(processed_data)
            most_probable_index = np.argmax(predictions)
            most_probable_class = LABELS_MAPPING.get(most_probable_index)
            sorted_classes = interpret_predictions(predictions)

            return Response({
                "most_probable_class": most_probable_class,
                "sorted_predictions": sorted_classes
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)        


"""class AnalyzeCsvView(views.APIView):

    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request, *args, **kwargs):
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            df = pd.read_csv(file)
            serializer = DataInputSerializer(data=df.to_dict(orient='records'), many=True)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            df = pd.DataFrame(serializer.validated_data)
            df = preprocessing(df)
            df = reshape_dataset_cnn(df)
            predictions = model.predict(df)

            predicted_classes = np.argmax(predictions, axis=1)

            # Map predicted classes to their labels
            predicted_labels = [LABELS_MAPPING.get(label) for label in predicted_classes]

            # Find the class with the highest probability
            most_probable_index = np.argmax(np.mean(predictions, axis=0))
            most_probable_class = LABELS_MAPPING.get(most_probable_index)

            # Create a sorted list of classes by probability
            sorted_indices = np.argsort(np.mean(predictions, axis=0))[::-1]
            sorted_classes_with_probs = [
                {"class": LABELS_MAPPING.get(index), "probability": float(np.mean(predictions, axis=0)[index])}
                for index in sorted_indices
            ]

            return Response({
                "most_probable_class": most_probable_class,
                "sorted_classes": sorted_classes_with_probs
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error("Error processing request: %s", str(e))
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
"""

        
"""class AnalyzeCsvView(views.APIView):
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request, *args, **kwargs):
        # Check if the request contains a 'file' field (indicating a file upload)
        if 'file' in request.FILES:
            file = request.FILES['file']

            try:
                # Read CSV data from the uploaded file
                df = pd.read_csv(file)
                
                # Process the data using DataInputSerializer
                serializer = DataInputSerializer(data=df.to_dict(orient='records'), many=True)
                if not serializer.is_valid():
                    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

                df = pd.DataFrame(serializer.validated_data)
                # Preprocess the input data
                processed_data = self.preprocess_input(df)
                predictions = model.predict(processed_data)

                # Get the most probable class
                most_probable_index = np.argmax(predictions)
                most_probable_class = LABELS_MAPPING.get(most_probable_index)

                # Sort the classes by probability (descending order)
                sorted_indices = np.argsort(predictions.ravel())[::-1]
                sorted_classes_with_probs = [
                    {"class": LABELS_MAPPING.get(index), "probability": float(predictions[index])}
                    for index in sorted_indices
                ]

                return Response({
                    "most_probable_class": most_probable_class,
                    "sorted_classes": sorted_classes_with_probs
                }, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        elif request.content_type == 'application/json':
            # JSON data received directly in the request body
            try:
                data = request.data
                # Create a QueryDict to handle JSON data similar to form data
                data_dict = QueryDict('', mutable=True)
                data_dict.update(data)
                
                
                # Convert validated data to DataFrame and perform further processing
                df = pd.DataFrame(data_dict)

                # Before model prediction
                print("Input shape:", df.shape)
                print("Input data:", df)
                
                processed_data = self.preprocess_input(df)
                print("Processed data:", processed_data)

                # After preprocessing and reshaping
                print("Processed input shape:", processed_data.shape)

                predictions = model.predict(processed_data)

                print("Predictions shape:", predictions.shape)
                print("Predictions:", predictions)

                # Get the most probable class
                most_probable_index = np.argmax(predictions)
                most_probable_class = LABELS_MAPPING.get(most_probable_index)

                return Response({
                    "most_probable_class": most_probable_class
                }, status=status.HTTP_200_OK)

            except Exception as e:
                logger.error("Error processing request: %s", str(e))
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        else:
            return Response({"error": "Invalid request format"}, status=status.HTTP_400_BAD_REQUEST)"""



"""def preprocessing(df: pd.DataFrame) -> np.ndarray:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[df < 0] = np.nan
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    return df

def reshape_dataset_cnn(x: np.ndarray) -> np.ndarray:
    required_padding = 81 - x.shape[1]
    padded_x = np.pad(x, ((0, 0), (0, required_padding)), 'constant', constant_values=0)
    reshaped_x = np.reshape(padded_x, (padded_x.shape[0], 9, 9, 1))
    return reshaped_x"""
