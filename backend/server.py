# 1. ייבוא הספריות
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 2. יצירת נתוני דוגמה
np.random.seed(42)
m2 = np.random.randint(50, 150, 100)  # שטח הדירות
price = m2 * 1000 + np.random.randint(-5000, 5000, 100)  # מחיר דירה בקשר ליניארי עם קצת רעש

# 3. המרת הנתונים לטבלה (DataFrame)
df = pd.DataFrame({'m2': m2, 'price': price})

# 4. חלוקת הנתונים לסט אימון וסט בדיקה
X = df[['m2']]  # משתנה עצמאי (שטח דירה)
y = df['price']  # משתנה תלוי (מחיר)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. יצירת מודל רגרסיה ליניארית
model = LinearRegression()

# 6. אימון המודל
model.fit(X_train, y_train)

# 7. בדיקה על נתוני הבדיקה
y_pred = model.predict(X_test)

# 8. הצגת התוצאות
plt.scatter(X_test, y_test, color='blue', label='נתונים אמיתיים')
plt.plot(X_test, y_pred, color='red', label='חיזוי המודל')
plt.xlabel('גודל דירה (מ"ר)')
plt.ylabel('מחיר דירה')
plt.legend()
plt.show()

# 9. הדפסת המקדמים של המודל
print(f"נוסחת החיזוי: מחיר = {model.coef_[0]:.2f} * גודל + {model.intercept_:.2f}")
