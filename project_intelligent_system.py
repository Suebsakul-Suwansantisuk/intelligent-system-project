import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
     page_title="Student Depression Prediction",
     page_icon="🧠",
     layout="wide",
     initial_sidebar_state="expanded"
)

# ── Sidebar navigation ─────────────────────────────────────────
st.sidebar.title("🧠 Student Depression")
st.sidebar.markdown("---")
page = st.sidebar.radio("เลือกหน้า", [
     "🏠 Home",
     "📘 ML Model",
     "📗 Neural Network",
])

# ══════════════════════════════════════════════════════════════
# หน้า 1 — อธิบายภาพรวม
# ══════════════════════════════════════════════════════════════

if page == "🏠 Home":
     st.title("Intelligent System Project")
     st.subheader("Page 1 : ML Model")
     st.markdown("ใช้ dataset ที่ชื่อว่า student_depression_dataset จาก Kaggle เพื่อทำนายว่านักศึกษามีเเนวโน้มว่าจะเป็นโรคซึมเศร้าหรือไม่ " \
     "โดยใช้โมเดล Ensemble Learning ที่รวม Random Forest, Logistic Regression และ K-Nearest Neighbors เข้าด้วยกัน")
     st.subheader("Page 2 : Neural Network")
     st.markdown("ใช้ dataset ที่ชื่อว่า diabetes จาก Kaggle เพื่อทำนายระดับความเสี่ยงต่อโรคเบาหวาน โดยใช้โมเดล Neural Network")

# ══════════════════════════════════════════════════════════════
# หน้า 2 — อธิบาย ML Model
# ══════════════════════════════════════════════
if page == "📘 ML Model":
     st.title("📘 Machine Learning Model")
     st.markdown("### Student Depression Prediction ด้วย Ensemble Learning")
     st.markdown("---")

     # 1. สำรวจข้อมูลเบื้องต้น
     st.header("1. สำรวจข้อมูลเบื้องต้น (Exploratory Data Analysis)")
     st.code("""
          data = pd.read_csv('student_depression_dataset.csv')
          display(data.head(15))
          print(data.dtypes)
          print(data.isnull().sum())
          print(f"จำนวน rows: {len(data)}")
          print(f"จำนวน columns: {len(data.columns)}")
     """, language="python")

     st.image("image_for_project/data1.png", caption="data 14 rows", use_container_width=True)
     col1, col2, col3, col4, col5 = st.columns([1, 2, 0.5, 2, 1])

     with col2:
          st.image("image_for_project/data2.png", caption="data types", use_container_width=True)
     with col4:
          st.image("image_for_project/data3.png", caption="check null values and quantity of columns and rows", use_container_width=True)

     st.subheader("สำรวจข้อมูลเบื้องต้นพบว่า")
     st.markdown("""
     - **Dataset:** student depression dataset จาก Kaggle
     - **จำนวนข้อมูล:** 27901 แถว, 18 columns
     - **Target variable:** `Depression` (0 = No , 1 =  Yes)
     - สำรวจข้อมูล 14 เเถวเเรก พบว่า ไม่มีค่า **NULL** เเต่บางค่ายังเป็น text จึงต้องนำไป encode 
     """)

     st.markdown("---")

     # 2. การเตรียมข้อมูล
     st.header("2. การเตรียมข้อมูล (Data Preparation)")
     
     st.subheader("การ Encode ข้อมูล")
     st.markdown("""
     columns ที่เป็น text ต้อง encode เป็นตัวเลขก่อน train โดยใช้ `LabelEncoder`:
     - `Gender`, `City`, `Profession`, `Sleep Duration`, `Dietary Habits`, `Degree`, `Have you ever had suicidal thoughts ?`, `Financial Stress`, `Family History of Mental Illness`
     """)
     st.code("""
     le = LabelEncoder()
     col_en = ['Gender','City','Profession','Sleep Duration','Dietary Habits','Degree','Have you ever had suicidal thoughts ?','Financial Stress','Family History of Mental Illness']
     for col in col_en:
     data[col] = le.fit_transform(data[col])
     """, language="python")
     
     st.image("image_for_project/data4.png", caption="data after encoding", use_container_width=True)

     st.subheader("การแบ่ง Train/Test")
     st.code("""
     X = data.drop(columns = ['id','Depression'])
     Y = data['Depression']
     X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state= 42)
     """, language="python")
 
     st.markdown("""
     - แบ่งข้อมูลเป็น 80% สำหรับ train และ 20% สำหรับ test โดยใช้ `train_test_split` จาก scikit-learn
     - X คือ features ทั้งหมด ยกเว้น `id` และ `Depression`
     - Y คือ target variable `Depression`
     """)

     st.markdown("---")

     # 3. การ train โมเดล
     st.header("3. การ train โมเดล (Model Training)")
     st.subheader("ทฤษฎีของอัลกอริทึมที่ใช้")
     col1, col2, col3 = st.columns(3)
     with col1:
          st.subheader("🌲 Random Forest")
          st.markdown("""
          สร้างต้นไม้การตัดสินใจหลายต้น (100 ต้น) แล้วโหวตเสียงข้างมาก
          - ไม่ต้อง Scale ข้อมูล
          - รับ mixed data ได้ดี
          - ดู feature importance ได้
          """)
     with col2:
          st.subheader("📈 Logistic Regression")
          st.markdown("""
          คำนวณเส้นแบ่งด้วยสมการทางคณิตศาสตร์ โดยปรับสมการให้เหมาะสมกับข้อมูลสูงสุด 1000 รอบ
          - ต้อง Scale ข้อมูลก่อน
          - อธิบายผลได้ง่าย
          - เหมาะกับ baseline
          """)
     with col3:
          st.subheader("👥 K-Nearest Neighbors")
          st.markdown("""
          วัดระยะห่างกับ k เพื่อนบ้านที่ใกล้ที่สุด แล้วโหวตผล
          - ต้อง Scale ข้อมูลก่อน (สำคัญมาก)
          - k = 5
          - เข้าใจง่าย อธิบายได้
          """)

     st.subheader("การนำทฤษฎีมาพัฒนาโมเดล")
     st.code("""
     RandomForest_pipe = Pipeline([
          ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
     ])

     LogisticRegression_pipe = Pipeline([
          ('scaler',StandardScaler()),
          ('lr',LogisticRegression(C=1.0, max_iter=1000, random_state = 42))
     ])

     KNN_pipe = Pipeline([
          ('scaler',StandardScaler()),
          ('knn',KNeighborsClassifier(n_neighbors=5))
     ])

     ensemble = VotingClassifier(estimators = [
          ('rf',RandomForest_pipe),
          ('lr',LogisticRegression_pipe),
          ('knn',KNN_pipe)
     ],voting = 'hard')

     ensemble.fit(X_train,Y_train)
     """, language="python")

     st.markdown("""
     - สร้าง pipeline เพราะเเต่ละโมเดลมีกระบวนการต่างกัน เช่น random forest ไม่ต้องมีการ scale ข้อมูล (ปรับให้แต่ละ feature มีค่าเฉลี่ย = 0 และส่วนเบี่ยงเบนมาตรฐาน = 1) เนื่องจากเเบ่งข้อมูลด้วยเงื่อนไขเเต่ knn เเละ logisticregression ต้อง scale  ข้อมูลก่อนจึงจะนำข้อมูลไปเทรน เพื่อป้องกัน feature dominance (ป้องกันไม่ให้ feature ที่มีตัวเลขใหญ่กว่าครอบงำโมเดล)
     - เมื่อสร้าง pipeline เเล้ว นำโมเดลทั้ง 3 มารวมกันใน `VotingClassifier` โดยใช้การโหวตเสียงข้างมาก (`voting='hard'`) เพื่อทำนายผลลัพธ์สุดท้าย
     - เทรนโมเดลด้วยข้อมูล train โดยใช้ `.fit()`"""
     )

     st.markdown("---")
 
     # 4. model evaluation
     st.header("4. การประเมินผลโมเดล (Model Evaluation)")
     st.code("""
     y_prediction = ensemble.predict(X_test)
     cm = confusion_matrix(Y_test,y_prediction)
     print(cm)
     print(classification_report(Y_test,y_prediction))
     """, language="python")

     col1, col2, col3 = st.columns([1, 1,1])
     with col2:
          st.image("image_for_project/data5.png", caption="Confusion Matrix and Classification Report", use_container_width=True)

     st.markdown("""
     นำข้อมูล test มาทำนายด้วยโมเดลที่เทรนเสร็จแล้ว แล้วประเมินผลลัพธ์ด้วย confusion matrix และ classification report เพื่อดูว่าโมเดลทำนายได้ดีแค่ไหน โดยดูจากค่า accuracy, precision, recall และ f1-score
     """)
     
     st.subheader("ผลลัพธ์ของโมเดล")
     col1, col2, col3 = st.columns(3)
     col1.metric("Accuracy", "83%")
     col2.metric("Macro F1-Score", "0.83")
     col3.metric("จำนวนข้อมูล Test", "5,581")
     
     st.markdown("---")

     # 5. อ้างอิง
     st.header("5. แหล่งอ้างอิง")
     st.markdown("""
     - Dataset: [Student Depression Dataset - Kaggle](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset)
     - scikit-learn Documentation: https://scikit-learn.org
     - Random Forest: Breiman, L. (2001). Random Forests. Machine Learning, 45, 5–32.
     """)

     st.markdown("---")

     # โหลด model
     model = joblib.load(os.path.join(BASE_DIR, 'model_ml.pkl'))

     st.title('Student Depression Prediction')
     st.write('กรอกข้อมูลเพื่อทำนายระดับ Depression')


     # ── Input form ──────────────────────────────────────────────
     col1, col2 = st.columns(2)

     with col1:
          st.subheader("ข้อมูลทั่วไป")
          gender       = st.selectbox("Gender", ["Female", "Male"])
          age          = st.slider("Age", 17, 35, 20)
          city         = st.selectbox("City", ["Bangalore", "Chennai", "Delhi",
                                                  "Hyderabad", "Kolkata", "Mumbai", "Pune"])
          profession   = st.selectbox("Profession", ["Student", "Working Professional"])
          degree       = st.selectbox("Degree", ["B.Com", "B.Ed", "BArch", "BCA",
                                                  "BE", "BHM", "BSc", "BTech",
                                                  "M.Com", "MA", "MBA", "MCA",
                                                  "MD", "ME", "MHM", "MSc", "MTech",
                                                  "PhD"])
          family_hist  = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

     with col2:
          st.subheader("ข้อมูลสุขภาพและการเรียน")
          academic_p   = st.slider("Academic Pressure", 0, 5, 3)
          work_p       = st.slider("Work Pressure", 0, 5, 2)
          cgpa         = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.1)
          study_sat    = st.slider("Study Satisfaction", 0, 5, 3)
          job_sat      = st.slider("Job Satisfaction", 0, 5, 3)
          sleep        = st.selectbox("Sleep Duration", ["Less than 5 hours",
                                                            "5-6 hours", "7-8 hours",
                                                            "More than 8 hours"])
          diet         = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
          suicidal     = st.selectbox("Have you ever had suicidal thoughts ?", ["No", "Yes"])
          work_study_h = st.slider("Work/Study Hours", 0, 12, 6)
          financial_s  = st.slider("Financial Stress", 0, 5, 3)

     # ── Encode mapping ───────────────────────────────────────────
     gender_enc   = {"Female": 0, "Male": 1}
     city_enc     = {"Bangalore": 0, "Chennai": 1, "Delhi": 2,
                    "Hyderabad": 3, "Kolkata": 4, "Mumbai": 5, "Pune": 6}
     prof_enc     = {"Student": 0, "Working Professional": 1}
     sleep_enc    = {"Less than 5 hours": 0, "5-6 hours": 1,
                    "7-8 hours": 2, "More than 8 hours": 3}
     diet_enc     = {"Healthy": 0, "Moderate": 1, "Unhealthy": 2}
     suicidal_enc = {"No": 0, "Yes": 1}
     family_enc   = {"No": 0, "Yes": 1}

     # degree มีเยอะ — ใช้ sorted เรียงตามตัวอักษรเหมือน LabelEncoder
     degree_list  = sorted(["B.Com","B.Ed","BArch","BCA","BE","BHM","BSc","BTech",
                         "M.Com","MA","MBA","MCA","MD","ME","MHM","MSc","MTech","PhD"])
     degree_enc   = {d: i for i, d in enumerate(degree_list)}

     input_df = pd.DataFrame([[
     gender_enc[gender], age, city_enc[city], prof_enc[profession],
     academic_p, work_p, cgpa, study_sat, job_sat,
     sleep_enc[sleep], diet_enc[diet], degree_enc[degree],
     suicidal_enc[suicidal], work_study_h, financial_s, family_enc[family_hist]
     ]], columns=[
     'Gender', 'Age', 'City', 'Profession', 'Academic Pressure', 'Work Pressure',
     'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration',
     'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
     'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness'
     ])

     # ── Predict ─────────────────────────────────────
     if st.button("🔍 ทำนาย", type="primary"):
          result = model.predict(input_df)[0]
     
     # Depression: 0 = No, 1 = Yes
          st.markdown("### ผลการทำนาย")
          if result == 0:
               st.success("## ผลลัพธ์: ไม่มีภาวะ Depression")
               st.markdown("ไม่พบความเสี่ยงภาวะซึมเศร้า")
          else:
               st.error("## ผลลัพธ์: มีภาวะ Depression")
               st.markdown("พบความเสี่ยงภาวะซึมเศร้า ควรปรึกษาผู้เชี่ยวชาญ")

# ══════════════════════════════════════════════════════════════
# หน้า 3 — อธิบาย Neural Network Model
# ══════════════════════════════════════════════════════════════
     
if page == "📗 Neural Network":
     st.title("📗 Neural Network")
     st.markdown("### Diabetes Prediction ด้วย Deep Learning")
     st.markdown("---")

     # 1. สำรวจข้อมูลเบื้องต้น
     st.header("1. สำรวจข้อมูลเบื้องต้น (Exploratory Data Analysis)")
     st.code("""
     data = pd.read_csv('diabetes.csv')
     display(data.head())
     print(data.dtypes)
     print(data.isnull().sum())
     print(f"จำนวน rows: {len(data)}")
     print(f"จำนวน columns: {len(data.columns)}")
     """, language="python")

     st.subheader("สำรวจข้อมูลเบื้องต้นพบว่า")
     st.markdown("""
     - **Dataset:** Pima Indians Diabetes Database จาก Kaggle
     - **จำนวนข้อมูล:** 768 แถว, 9 columns
     - **Target variable:** `Outcome` (0 = No Diabetes, 1 = Diabetes)
     - ไม่มีค่า NULL และทุก column เป็นตัวเลขอยู่แล้ว ไม่ต้อง encode
     """)

     st.markdown("---")

     # 2. การเตรียมข้อมูล
     st.header("2. การเตรียมข้อมูล (Data Preparation)")

     st.subheader("การ Scale ข้อมูล")
     st.markdown("""
     Neural Network ต้อง scale ข้อมูลทุกครั้ง โดยใช้ `StandardScaler`
     ปรับให้แต่ละ feature มีค่าเฉลี่ย = 0 และส่วนเบี่ยงเบนมาตรฐาน = 1
     เพื่อป้องกัน feature dominance และช่วยให้โมเดลเรียนรู้ได้เร็วขึ้น
     """)
     st.code("""
     scaler = StandardScaler()
     X = scaler.fit_transform(X)
     """, language="python")

     st.subheader("การแบ่ง Train/Test")
     st.code("""
     X = df.drop(columns=['Outcome'])
     y = df['Outcome']

     X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42
     )
     """, language="python")

     st.markdown("""
     - แบ่งข้อมูลเป็น 80% สำหรับ train และ 20% สำหรับ test
     - X คือ features ทั้งหมด ยกเว้น `Outcome`
     - y คือ target variable `Outcome`
     """)

     st.markdown("---")

     # 3. การ train โมเดล
     st.header("3. การ train โมเดล (Model Training)")

     st.subheader("ทฤษฎีของ Neural Network")
     st.markdown("""
     Neural Network ประกอบด้วย layers หลายชั้นที่เชื่อมต่อกัน แต่ละ layer
     เรียนรู้ pattern จากข้อมูลในระดับที่ซับซ้อนขึ้นเรื่อยๆ
     """)

     col1, col2, col3 = st.columns(3)
     with col1:
          st.subheader("🔵 Input Layer")
          st.markdown("""
          รับ features ทั้งหมดเข้ามา
          - 8 neurons (8 features)
          - Pregnancies, Glucose, BloodPressure
          - SkinThickness, Insulin, BMI
          - DiabetesPedigreeFunction, Age
          """)
     with col2:
          st.subheader("🟡 Hidden Layers")
          st.markdown("""
          เรียนรู้ pattern จากข้อมูล
          - Layer 1: 16 neurons, ReLU
          - Layer 2: 8 neurons, ReLU
          - Dropout 0.2 ป้องกัน overfit
          - ReLU: f(x) = max(0, x)
          """)
     with col3:
          st.subheader("🟢 Output Layer")
          st.markdown("""
          ทำนายผลลัพธ์สุดท้าย
          - 1 neuron (Binary)
          - Sigmoid activation
          - output = 0-1 (probability)
          - threshold 0.5 → Yes/No
          """)

     st.subheader("โครงสร้างโมเดล")
     st.code("""
     model = Sequential([
     Dense(16, activation='relu', input_shape=(8,)),
     Dropout(0.2),
     Dense(8, activation='relu'),
     Dropout(0.2),
     Dense(1, activation='sigmoid')
     ])

     model.compile(
     optimizer='adam',
     loss='binary_crossentropy',
     metrics=['accuracy']
     )
     """, language="python")

     st.subheader("การเลือก Activation Function")
     st.markdown("""
     - **Hidden layers → ReLU** เพราะคำนวณเร็ว ป้องกัน vanishing gradient ได้ดี
     - **Output layer → Sigmoid** เพราะเป็น Binary Classification ต้องการค่า 0-1
     """)

     st.subheader("การ Train โมเดล")
     st.code("""
     early_stop = EarlyStopping(
     monitor='val_loss',
     patience=15,
     restore_best_weights=True
     )

     history = model.fit(
     X_train, y_train,
     epochs=200,
     batch_size=16,
     validation_split=0.1,
     callbacks=[early_stop]
     )
     """, language="python")

     st.markdown("""
     - ใช้ **EarlyStopping** หยุด train อัตโนมัติเมื่อ val_loss ไม่ดีขึ้นติดกัน 15 epochs
     - ใช้ **Dropout** สุ่มปิด neuron 20% ในแต่ละรอบเพื่อป้องกัน overfit
     - **Adam optimizer** ปรับ learning rate อัตโนมัติ เหมาะกับงานทั่วไป
     - **binary_crossentropy** เหมาะกับ Binary Classification
     """)

     st.markdown("---")

     # 4. model evaluation
     st.header("4. การประเมินผลโมเดล (Model Evaluation)")
     st.code("""
     y_pred_prob = model.predict(X_test)
     y_pred = (y_pred_prob > 0.5).astype(int)

     print(confusion_matrix(y_test, y_pred))
     print(classification_report(y_test, y_pred,
          target_names=['No Diabetes', 'Diabetes']))
     """, language="python")

     st.markdown("""
     Neural Network output เป็น probability (0.0 - 1.0) จึงต้องแปลงเป็น class ก่อน
     โดยใช้ threshold 0.5 ถ้า probability > 0.5 → Diabetes, ถ้าน้อยกว่า → No Diabetes
     """)

     st.subheader("ผลลัพธ์ของโมเดล")
     col1, col2, col3 = st.columns(3)
     col1.metric("Accuracy", "77%")
     col2.metric("Architecture", "3 Layers")
     col3.metric("จำนวนข้อมูล Test", "154")

     st.markdown("---")

     # 5. อ้างอิง
     st.header("5. แหล่งอ้างอิง")
     st.markdown("""
     - Dataset: [Pima Indians Diabetes Database - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
     - TensorFlow / Keras Documentation: https://keras.io
     - Dropout: Srivastava et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
     - Adam Optimizer: Kingma & Ba (2014). Adam: A Method for Stochastic Optimization.
     """)

     st.markdown("---")

     # ── ส่วนทดสอบโมเดล ──────────────────────────────────────────

     try:
          nn_model = joblib.load(os.path.join(BASE_DIR, 'nn_model.pkl'))
          nn_scaler = joblib.load(os.path.join(BASE_DIR, 'scaler_nn.pkl'))
     except:
          st.error("ไม่พบไฟล์ nn_model.pkl หรือ scaler_nn.pkl")
          st.stop()

     st.title("Diabetes Prediction")
     st.write("กรอกข้อมูลเพื่อทำนายความเสี่ยง Diabetes")

     col1, col2 = st.columns(2)

     with col1:
          st.subheader("ข้อมูลทั่วไป")
          pregnancies = st.slider("Pregnancies", 0, 17, 3)
          age         = st.slider("Age", 21, 81, 33)
          bmi         = st.slider("BMI", 0.0, 67.0, 32.0, step=0.1)
          dpf         = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01)

     with col2:
          st.subheader("ข้อมูลสุขภาพ")
          glucose      = st.slider("Glucose", 0, 200, 120)
          blood_p      = st.slider("Blood Pressure", 0, 122, 70)
          skin_t       = st.slider("Skin Thickness", 0, 99, 20)
          insulin      = st.slider("Insulin", 0, 846, 80)

     input_nn = pd.DataFrame([[
          pregnancies, glucose, blood_p, skin_t,
          insulin, bmi, dpf, age
     ]], columns=[
          'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
     ])

     # scale ก่อน predict
     input_scaled = nn_scaler.transform(input_nn)

     if st.button("🔍 ทำนาย", type="primary"):
          prob = nn_model.predict_proba(input_scaled)[0][1]
          result = nn_model.predict(input_scaled)[0]

          st.markdown("### ผลการทำนาย")
          if result == 0:
               st.success(f"## ผลลัพธ์: ไม่มีความเสี่ยง Diabetes 🟢")
               st.markdown(f"ความเสี่ยง: {prob*100:.1f}%")
          else:
               st.error(f"## ผลลัพธ์: มีความเสี่ยง Diabetes 🔴")
               st.markdown(f"ความเสี่ยง: {prob*100:.1f}% — ควรปรึกษาแพทย์")