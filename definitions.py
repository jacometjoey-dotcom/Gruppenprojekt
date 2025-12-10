import streamlit as st


#removing outliers using the iqr method (ML part)
#video: https://www.youtube.com/watch?v=9jYqZS142mg

def removing_outliers(v_df, column):
    Q1 = v_df[column].quantile(0.25)
    Q3 = v_df[column].quantile(0.75)
    IQR = Q3 - Q1 #calculating the interquartile (iqr) range (where the middle 50% of the values lie, the dispersion)
    lower_bound = Q1 - 1.5 * IQR #downard outliers
    upper_bound = Q3 + 1.5 * IQR #upward outliers 
    v_df_clean = v_df[(v_df[column] >= lower_bound) & (v_df[column] <= upper_bound)] #keeping the values between the limits, everything else is removed 
    return v_df_clean 


#A: The next section is relevant for the Calculation section of the app, here Ivan created a whole lot of functions to cope with python shortcomings regarding number formatting and input validation

# The following helper functions are used to format and validate user inputs and outputs related to money, percentages, and years.

def format_thousands_ch(num: float, decimals: int = 0) -> str:
    s = f"{num:,.{decimals}f}"
    s = s.replace(",", "_").replace(".", ",").replace("_", "’")
    if decimals == 0:
        s = s.split(",")[0]
    return s


def show_money(n: float | None) -> str:
    if n is None:
        return ""
    return f"{format_thousands_ch(n, 0)} CHF"


def show_percent(ratio: float | None) -> str:
    if ratio is None:
        return ""
    return f"{ratio*100:.2f}%"


def show_years(y: float | None) -> str:
    if y is None:
        return ""
    return f"{y:.2f} years"


def strip_common_bits(text: str) -> str:
    s = str(text or "").strip()
    s = s.replace(" ", "").replace("’", "").replace("'", "")
    s = s.replace("CHF", "").replace("chf", "")
    s = s.replace("YEARS", "").replace("years", "").replace("Year", "").replace("year", "")
    s = s.replace("%", "")
    return s


def check_number_input(label: str, text: str):
    if text is None or str(text).strip() == "":
        return None, f"{label}: is required and must be a number."
    s = strip_common_bits(text)
    s = s.replace(",", "")
    try:
        return float(s), None
    except ValueError:
        return None, f"{label}: must be a valid number (e.g., 3’700’000 or 3700000)."


def check_years_input(label: str, text: str):
    if text is None or str(text).strip() == "":
        return None, f"{label}: is required and must be a positive number."
    s = strip_common_bits(text)
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        val = float(s)
    except ValueError:
        return None, f"{label}: must be a valid number (e.g., 1.50)."
    if val <= 0:
        return None, f"{label}: must be greater than 0."
    return val, None


def check_percent_input(label: str, text: str):
    if text is None or str(text).strip() == "":
        return None, f"{label}: is required and must be a percentage."
    s = strip_common_bits(text)
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    s = s.replace(",", "")
    try:
        val = float(s)
    except ValueError:
        return None, f"{label}: must be a valid percentage (e.g., 6.25 or 6.25%)."
    ratio = val if val <= 1 else val / 100.0
    if ratio < 0:
        return None, f"{label}: cannot be negative."
    if ratio > 1:
        return None, f"{label}: cannot exceed 100%."
    return ratio, None


def check_crowdfunder_count(label: str, text: str):
    if text is None or str(text).strip() == "":
        return None, f"{label}: is required and must be a whole number from 1 to 19."
    s = strip_common_bits(text)
    if "." in s or "," in s:
        return None, f"{label}: must be an integer (no decimals)."
    if not s.isdigit():
        return None, f"{label}: must be an integer (e.g., 7)."
    val = int(s)
    if not (1 <= val <= 19):
        return None, f"{label}: must be between 1 and 19."
    return val, None


def touchup_money_key(key: str):
    raw = st.session_state.get(key, "")
    if raw is None or str(raw).strip() == "":
        return
    val, err = check_number_input("", raw)
    if err is None:
        st.session_state[key] = show_money(val)


def touchup_percent_key(key: str):
    raw = st.session_state.get(key, "")
    if raw is None or str(raw).strip() == "":
        return
    ratio, err = check_percent_input("", raw)
    if err is None:
        st.session_state[key] = show_percent(ratio)


def touchup_years_key(key: str):
    raw = st.session_state.get(key, "")
    if raw is None or str(raw).strip() == "":
        return
    years_val, err = check_years_input("", raw)
    if err is None:
        st.session_state[key] = show_years(years_val)


def show_readonly(label: str, value: str):
    if value:
        st.text_input(label, value=value, disabled=True)


def show_highlight_green(label: str, value: str):
    if value:
        st.success(f"{label}: {value}")