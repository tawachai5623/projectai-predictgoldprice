import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# โหลดข้อมูล
df = pd.read_csv("goldstock-v2.csv")
df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.index = df['Date']

st.title("Gold Price Prediction")
# แสดงกราฟราคาทองคำทั้งหมด
st.subheader("Gold Price Over All Years")
fig_all, ax_all = plt.subplots(figsize=(10, 6))
ax_all.plot(df['Date'], df['Close/Last'], label="Gold Price", color='blue')
ax_all.set_xlabel('Date')
ax_all.set_ylabel('Price (USD)')
ax_all.set_title("Gold Price Over Time")
ax_all.legend()

# แสดงกราฟใน Streamlit
st.pyplot(fig_all)

# เพิ่มให้ผู้ใช้สามารถเลือกปี
st.subheader("Gold Price Prediction by Year")
selected_year = st.selectbox('Select year', options=sorted(df['Date'].dt.year.unique()))

# กรองข้อมูลตามปีที่เลือก
yearly_data = df[df['Date'].dt.year == selected_year]

# การแสดงกราฟราคาทองคำต่อปี
st.subheader(f"Gold Price for Year {selected_year}")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(yearly_data['Date'], yearly_data['Close/Last'], label="Gold Price", color='gold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.set_title(f"Gold Price in {selected_year}")
ax.legend()

# แสดงกราฟใน Streamlit
st.pyplot(fig)

# แสดงข้อมูลในตาราง
st.write("Data for the selected year:")
st.dataframe(yearly_data[['Date', 'Close/Last']])

# ฟังก์ชันสำหรับทำนายราคาในอนาคต (เช่น 3 ปีข้างหน้า)
def predict_future_prices(df, num_predictions=36):
    # สเกลข้อมูล
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Close/Last']])
    
    # เตรียมข้อมูลสำหรับการฝึกโมเดล
    x_train_data = []
    y_train_data = []
    
    for i in range(60, len(df_scaled)):
        x_train_data.append(df_scaled[i-60:i, 0])  # เลือกข้อมูล 60 วันก่อนหน้าเป็น input
        y_train_data.append(df_scaled[i, 0])  # ข้อมูลในวันถัดไปเป็น target

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
    
    # Reshape สำหรับ LSTM
    x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    # สร้างโมเดล LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')

    # ฝึกโมเดล
    model.fit(x_train_data, y_train_data, epochs=100, batch_size=32, verbose=0)

    # ทำนายอนาคต 36 เดือน
    last_60_days = df_scaled[-60:]
    X_future = np.array(last_60_days).reshape(1, 60, 1)

    predicted_future_prices = []

    for i in range(num_predictions):
        predicted_price = model.predict(X_future)
        predicted_future_prices.append(predicted_price[0, 0])

        # แก้ไขการเพิ่ม predicted_price
        X_future = np.append(X_future[:, 1:, :], np.reshape(predicted_price, (1, 1, 1)), axis=1)

    predicted_future_prices = np.array(predicted_future_prices)
    predicted_future_prices = scaler.inverse_transform(predicted_future_prices.reshape(-1, 1))

    return predicted_future_prices


# แสดงผลการทำนายราคาในอนาคต 3 ปี
if st.button(f"Predict Gold Prices for the next 3 years from {selected_year}"):
    predicted_prices = predict_future_prices(df[['Close/Last']])
    future_dates = pd.date_range(start=yearly_data.index[-1], periods=len(predicted_prices), freq='M')

    future_df = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted Gold Price'])

    # คำนวณช่วงปีเริ่มต้นและสิ้นสุด
    start_year = yearly_data.index[-1].year  # ปีเริ่มต้นจากข้อมูลจริง
    end_year = start_year + 3  # บวก 3 ปีสำหรับการทำนาย

    # แสดงกราฟราคาทองคำในอนาคต
    st.subheader(f"Gold Price Prediction for {start_year}-{end_year}")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(future_df.index, future_df['Predicted Gold Price'], label="Predicted Gold Prices", color='orange')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price (USD)')
    ax2.set_title(f'Gold Price Prediction ({start_year}-{end_year})')
    ax2.legend()

    st.pyplot(fig2)
