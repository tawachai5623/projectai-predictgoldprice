import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

# ตั้งค่า title
st.title("Gold Price Prediction with MLP")

# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv("goldstock-v2.csv")
df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.set_index('Date', inplace=True)

# รับค่าจำนวนเดือนที่ต้องการทำนายจากผู้ใช้
num_predictions_input = st.text_input('Enter the number of months to predict:', '')

# ตรวจสอบและแปลงค่าที่กรอกเป็นจำนวนเดือน
if num_predictions_input:
    try:
        num_predictions = int(num_predictions_input)
    except ValueError:
        st.error("Please enter a valid number.")
        num_predictions = 0  # ตั้งค่าเริ่มต้นถ้ากรอกไม่ถูกต้อง
else:
    num_predictions = 0  # ตั้งค่าเริ่มต้นถ้ากรอกว่าง

if num_predictions > 0:
    # การเตรียมข้อมูลสำหรับการสร้างโมเดล
    data = df.sort_index(ascending=True, axis=0)
    new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close/Last'])

    for i in range(0, len(data)):
        new_dataset["Date"][i] = data.index[i]  # ใช้ index แทนการเข้าถึง 'Date'
        new_dataset["Close/Last"][i] = data["Close/Last"][i]

    new_dataset.index = new_dataset.Date
    new_dataset.drop("Date", axis=1, inplace=True)

    # การแบ่งข้อมูลเป็นชุดฝึก (Training Set) และชุดทดสอบ (Test Set)
    final_dataset = new_dataset.values
    train_data = final_dataset[0:1255, :]
    valid_data = final_dataset[1256:, :]

    # การปรับสเกลข้อมูล (Scaling)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)

    # เตรียมข้อมูลสำหรับการฝึกโมเดล
    x_train_data = []
    y_train_data = []

    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i-60:i, 0])  # เลือกข้อมูล 60 วันก่อนหน้าเป็น input
        y_train_data.append(scaled_data[i, 0])  # ข้อมูลในวันถัดไปเป็น target

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    # ปรับข้อมูลให้เป็นแบบ 2 มิติ (ไม่มี sequence dimension)
    x_train_data = x_train_data.reshape(x_train_data.shape[0], x_train_data.shape[1])

    # ### การสร้างและฝึกโมเดล MLP ###
    mlp_model = Sequential()
    mlp_model.add(Dense(units=50, activation='relu', input_shape=(x_train_data.shape[1],)))
    mlp_model.add(Dense(units=50, activation='relu'))
    mlp_model.add(Dense(units=1))  # Output layer

    # คอมไพล์โมเดล
    mlp_model.compile(optimizer='adam', loss='mean_squared_error')

    # ฝึกโมเดล
    mlp_model.fit(x_train_data, y_train_data, epochs=150, batch_size=32)

    # เตรียมข้อมูลทดสอบสำหรับโมเดล MLP
    inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60:].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)

    X_test = []
    for i in range(60, inputs_data.shape[0]):
        X_test.append(inputs_data[i-60:i, 0])
    X_test = np.array(X_test)

    # ปรับข้อมูลให้เป็นแบบ 2 มิติสำหรับ MLP (ไม่มี sequence dimension)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # ใช้ MLP model ในการทำนาย
    prediction_closing = mlp_model.predict(X_test)
    prediction_closing = scaler.inverse_transform(prediction_closing.reshape(-1, 1))

    # การแสดงผลกราฟ
    train_data = new_dataset[:1255]
    valid_data = new_dataset[1256:]
    valid_data['Predictions'] = prediction_closing

    # แสดงกราฟการทำนาย
    st.subheader("Training and Prediction Data")
    plt.figure(figsize=(16, 8))
    plt.plot(train_data["Close/Last"], label="Training Data")
    plt.plot(valid_data["Close/Last"], label="Actual Data")
    plt.plot(valid_data["Predictions"], label="Predicted Data", linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Training vs Actual vs Predicted')
    plt.legend()
    st.pyplot(plt)

    # เตรียมข้อมูลอินพุตล่าสุด (60 วันที่ผ่านมา)
    last_60_days = new_dataset[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))

    X_future = []
    X_future.append(last_60_days_scaled)

    # คาดการณ์อนาคต
    predicted_future_prices = []

    for i in range(num_predictions):
        X_future_input = np.array(X_future[-1]).reshape(1, 60)
        predicted_price = mlp_model.predict(X_future_input)
        predicted_future_prices.append(predicted_price[0, 0])

        # เพิ่มค่าทำนายเข้าไปในชุดข้อมูล X_future สำหรับการทำนายครั้งถัดไป
        X_future_scaled = np.append(X_future[-1][1:], predicted_price)
        X_future.append(X_future_scaled.reshape(60, 1))

    # แปลงค่าทำนายกลับมาเป็นค่าราคาจริง (inverse scaling)
    predicted_future_prices = np.array(predicted_future_prices)
    predicted_future_prices = scaler.inverse_transform(predicted_future_prices.reshape(-1, 1))

    # สร้างวันที่สำหรับการคาดการณ์ เริ่มต้นจากปี 2024
    future_dates = pd.date_range(start='2024-07-02', periods=num_predictions, freq='M')

    # สร้าง DataFrame สำหรับแสดงผลการคาดการณ์
    future_df = pd.DataFrame(predicted_future_prices, index=future_dates, columns=['Predicted Gold Price'])

    # แสดงกราฟการคาดการณ์ที่ต่อเนื่องจากข้อมูลในอดีต
    st.subheader("Historical and Predicted Gold Prices")

    # สร้าง DataFrame สำหรับการรวมข้อมูลอดีตและคาดการณ์
    combined_df = pd.concat([df[['Close/Last']], future_df])

    # แสดงกราฟ
    plt.figure(figsize=(16, 8))
    plt.plot(combined_df.index, combined_df["Close/Last"], label="Historical Gold Prices")
    plt.plot(future_df.index, future_df['Predicted Gold Price'], label="Predicted Gold Prices (2024-2026)", color='orange')
    plt.xlabel('Date')
    plt.ylabel('Gold Price')
    plt.title('Gold Price Prediction for 2024-2026')
    plt.legend()
    st.pyplot(plt)  # ใช้ st.pyplot() เพื่อแสดงกราฟใน Streamlit

    # ใช้ cross validation ในการประเมินโมเดล MLP
    st.subheader("Cross Validation Results")
    n_folds = 5
    kf = KFold(n_splits=n_folds)
    X = x_train_data  # ข้อมูลอินพุต
    y = y_train_data  # ข้อมูลเอาท์พุต

    # กำหนดค่าตัวแปรสำหรับการบันทึกค่าเฉลี่ยของผลลัพธ์
    fold_scores = []

    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        # สร้างโมเดล MLP ใหม่ในแต่ละ fold
        model = Sequential()
        model.add(Dense(units=50, activation='relu', input_shape=(X_train_fold.shape[1],)))
        model.add(Dense(units=50, activation='relu'))
        model.add(Dense(1))  # Output layer

        # คอมไพล์โมเดล
        model.compile(optimizer='adam', loss='mean_squared_error')

        # ฝึกโมเดล
        model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)

        # ประเมินโมเดลบนชุด validation
        score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        fold_scores.append(score)

    # แสดงค่าเฉลี่ยของผลลัพธ์จากทุก fold
    mean_score = np.mean(fold_scores)
    st.write(f"Mean cross-validation loss: {mean_score:.4f}")
else:
    st.write("Please enter a number of months to predict.")
