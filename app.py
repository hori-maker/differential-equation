import streamlit as st
import numpy as np
from scipy.integrate import odeint
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------
# 設定 & デザイン
# -------------------------------------------------------
st.set_page_config(page_title="微分方程式シミュレーター", layout="wide")
st.markdown("""<style>.block-container { padding-top: 2rem; padding-bottom: 2rem; }</style>""", unsafe_allow_html=True)

# -------------------------------------------------------
# サイドバー：分野別メニュー
# -------------------------------------------------------
st.sidebar.header("📚 分野を選択")

field = st.sidebar.radio(
    "",
    ("🐰 生物 (生態系)", "💊 医療 (感染・薬)", "💰 経済 (マーケティング)", "🏎️ 物理 (自然法則)", "💘 番外編 (恋愛)", "🪐 カオス (三体問題)")
)

st.sidebar.markdown("---")
st.sidebar.header("🎮 パラメータ設定")

# -------------------------------------------------------
# 各モデルの定義
# 戻り値: (df, 解説文, 数式(LaTeX), Y軸範囲)
# -------------------------------------------------------

# === 🐰 生物分野 ===
def run_lotka_volterra():
    alpha = st.sidebar.slider("ウサギの繁殖力 (alpha)", 0.1, 2.0, 1.0)
    beta = st.sidebar.slider("食べられる率 (beta)", 0.1, 1.0, 0.1)
    delta = st.sidebar.slider("キツネの増殖効率 (delta)", 0.01, 0.5, 0.075)
    gamma = st.sidebar.slider("キツネの餓死率 (gamma)", 0.1, 1.0, 0.5)
    
    t = np.linspace(0, 100, 200)
    X0 = [20, 5] 

    def model(X, t):
        x, y = X
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        return dxdt, dydt

    y = odeint(model, X0, t)
    df = pd.DataFrame(y, columns=['ウサギ (x)', 'キツネ (y)'])
    df['Time'] = t
    
    latex = r"""
    \begin{cases}
    \frac{dx}{dt} = \alpha x - \beta xy \\
    \frac{dy}{dt} = \delta xy - \gamma y
    \end{cases}
    """
    desc = """
    **ロトカ・ヴォルテラの方程式**
    *   $x$: 被食者（ウサギ）、$y$: 捕食者（キツネ）
    *   ウサギは勝手に増えます($\alpha x$)が、キツネに出会うと減ります($-\beta xy$)。
    *   キツネはウサギに出会うと増えます($\delta xy$)が、放っておくと死にます($-\gamma y$)。
    """
    return df, desc, latex, None

def run_logistic():
    r = st.sidebar.slider("増殖率 (r)", 0.1, 1.0, 0.2)
    K = st.sidebar.slider("環境収容力 (K)", 50, 200, 100)
    N0 = st.sidebar.number_input("初期個体数", value=10)
    
    t = np.linspace(0, 100, 100)
    y = odeint(lambda N, t: r * N * (1 - N/K), N0, t).flatten()
    df = pd.DataFrame({'Time': t, '個体数 (N)': y})
    
    latex = r"\frac{dN}{dt} = rN \left(1 - \frac{N}{K}\right)"
    desc = """
    **ロジスティック方程式**
    *   人口爆発を防ぐ有名な式です。
    *   カッコの中の $(1 - N/K)$ が**「混雑ブレーキ」**です。
    *   人口 $N$ が定員 $K$ に近づくと、ブレーキがかかって増加率が 0 になります。
    """
    return df, desc, latex, [0, 250]


# === 💊 医療分野 ===
def run_sir():
    beta = st.sidebar.slider("感染率 (beta)", 0.0, 1.0, 0.3)
    gamma = st.sidebar.slider("回復率 (gamma)", 0.0, 1.0, 0.1)
    
    N = 1000
    t = np.linspace(0, 160, 160)
    y = odeint(lambda z, t: [-beta*z[0]*z[1]/N, beta*z[0]*z[1]/N - gamma*z[1], gamma*z[1]], [N-1, 1, 0], t)
    df = pd.DataFrame(y, columns=['未感染 (S)', '感染中 (I)', '回復済 (R)'])
    df['Time'] = t
    
    latex = r"""
    \begin{cases}
    \frac{dS}{dt} = -\beta \frac{SI}{N} \\
    \frac{dI}{dt} = \beta \frac{SI}{N} - \gamma I \\
    \frac{dR}{dt} = \gamma I
    \end{cases}
    """
    desc = """
    **SIRモデル (Kermack–McKendrick theory)**
    *   SとIが出会う確率($S \times I$)に比例して感染が進みます。
    *   同時に、一定の割合($\gamma I$)で人は治っていきます。
    *   **$dI/dt$ (感染者の増減)** がマイナスになれば、流行は収束します。
    """
    return df, desc, latex, [0, 1050]

def run_drug():
    ka = st.sidebar.slider("吸収速度 (ka)", 0.1, 2.0, 0.5)
    ke = st.sidebar.slider("排出速度 (ke)", 0.05, 1.0, 0.2)
    
    t = np.linspace(0, 24, 100)
    def model(y, t):
        G, B = y
        dGdt = -ka * G
        dBdt = ka * G - ke * B
        return dGdt, dBdt

    y = odeint(model, [100, 0], t)
    df = pd.DataFrame(y, columns=['胃の中の薬量', '血中濃度'])
    df['Time'] = t
    
    latex = r"""
    \begin{cases}
    \frac{dG}{dt} = -k_a G \\
    \frac{dB}{dt} = k_a G - k_e B
    \end{cases}
    """
    desc = """
    **薬物動態 (1-コンパートメントモデル)**
    *   $G$: 胃に残っている薬、$B$: 血液中の薬
    *   胃からはどんどん減り($-k_a G$)、その分が血液に入ります。
    *   血液からは尿として排出($-k_e B$)されます。
    *   この連立方程式を解くことで、「食後何時間で効き目がピークになるか」が分かります。
    """
    return df, desc, latex, [0, 100]


# === 💰 経済分野 ===
def run_bass():
    p = st.sidebar.slider("広告効果 (p)", 0.000, 0.05, 0.003, format="%.3f")
    q = st.sidebar.slider("口コミ効果 (q)", 0.0, 1.0, 0.4)
    M = 5000
    
    t = np.linspace(0, 50, 50)
    y = odeint(lambda N, t: (p + q * N / M) * (M - N), 0, t).flatten()
    speed = (p + q * y / M) * (M - y)
    
    df = pd.DataFrame({'Time': t, '累計売上 (N)': y, '売上の勢い (dN/dt)': speed})
    
    latex = r"\frac{dN}{dt} = \left( p + \frac{q}{M}N \right) (M - N)"
    desc = """
    **バス拡散モデル (Bass Diffusion Model)**
    *   $N$: すでに買った人の数、$M$: 全体の市場規模
    *   $(M-N)$: まだ買っていない人の数
    *   買う動機は2つあります。
        1.  $p$: 広告を見て独自に買う（イノベーター）
        2.  $\frac{q}{M}N$: すでに持っている人の数に影響されて買う（フォロワー）
    """
    return df, desc, latex, [0, 6000]


# === 🏎️ 物理分野 ===
def run_spring():
    k = st.sidebar.slider("バネ定数 (k)", 0.1, 5.0, 1.0)
    c = st.sidebar.slider("抵抗係数 (c)", 0.0, 1.0, 0.1)
    m = 1.0
    
    t = np.linspace(0, 50, 200)
    y = odeint(lambda X, t: [X[1], -(c/m)*X[1] - (k/m)*X[0]], [5.0, 0.0], t)
    df = pd.DataFrame(y, columns=['位置 (x)', '速度 (v)'])
    df['Time'] = t
    
    latex = r"m \frac{d^2 x}{dt^2} = -c \frac{dx}{dt} - kx"
    desc = """
    **減衰振動 (Damped Harmonic Oscillator)**
    *   運動方程式 $F=ma$ そのものです。
    *   力 $F$ には、元に戻ろうとするバネの力($-kx$)と、動きを邪魔する空気抵抗($-cv$)の2つが働いています。
    *   抵抗 $c=0$ なら永遠に動き続け、抵抗が大きいと振動せずに止まります。
    """
    return df, desc, latex, [-6, 6]

def run_cooling():
    k = st.sidebar.slider("冷却定数 (k)", 0.01, 0.20, 0.05)
    T_env = 20
    T_init = 90
    
    t = np.linspace(0, 100, 100)
    y_analytic = T_env + (T_init - T_env) * np.exp(-k * t)
    df = pd.DataFrame({'Time': t, '温度 (T)': y_analytic})
    
    latex = r"\frac{dT}{dt} = -k (T - T_{env})"
    desc = """
    **ニュートンの冷却法則**
    *   温度の変化スピード $dT/dt$ は、「周りとの温度差」に比例します。
    *   数式を変形（変数分離）して積分すると、右辺に $\int -k dt$ が出るため、解には $e^{-kt}$ （指数関数）が現れます。
    """
    return df, desc, latex, [0, 100]


# === 💘 番外編 (恋愛) ===
def run_love():
    st.sidebar.markdown("##### 性格パラメータ")
    a = st.sidebar.slider("ロミオの情熱 (a)", -1.0, 1.0, 0.5)
    b = st.sidebar.slider("ジュリエットの情熱 (b)", -1.0, 1.0, -0.5)

    t = np.linspace(0, 20, 200)
    def model(X, t):
        R, J = X
        dRdt = a * J 
        dJdt = b * R 
        return dRdt, dJdt

    y = odeint(model, [1, 1], t)
    df = pd.DataFrame(y, columns=['ロミオ (R)', 'ジュリエット (J)'])
    df['Time'] = t
    
    latex = r"""
    \begin{cases}
    \frac{dR}{dt} = a J \\
    \frac{dJ}{dt} = b R
    \end{cases}
    """
    desc = """
    **恋愛の力学系 (Strogatz Model)**
    *   $dR/dt$: ロミオの気持ちの変化率は、ジュリエットの気持ち($J$)に比例する。
    *   **パラメータの意味:**
        *   $a > 0$: 相手が好きだと盛り上がる（純粋）
        *   $b < 0$: 相手が好きすぎると冷める（天邪鬼）
    *   物理のバネと同じ式になるため、感情も「振動」します。
    """
    return df, desc, latex, [-3, 3]


# === 🪐 カオス (三体問題) ===
def run_three_body():
    st.sidebar.info("再生ボタン(▶)で動きます")
    
    # ★スライダーを個別に復活させました！★
    st.sidebar.subheader("3つの星の質量")
    m1 = st.sidebar.slider("青い星 (m1)", 1.0, 20.0, 10.0)
    m2 = st.sidebar.slider("赤い星 (m2)", 1.0, 20.0, 10.0)
    m3 = st.sidebar.slider("緑の星 (m3)", 1.0, 20.0, 10.0)
    
    t = np.linspace(0, 20, 300); G = 1.0
    state0 = [0.97, -0.24, 0.46, 0.43, -0.97, 0.24, 0.46, 0.43, 0, 0, -2*0.46, -2*0.43]

    def model(state, t, m1, m2, m3):
        r1, v1 = state[0:2], state[2:4]; r2, v2 = state[4:6], state[6:8]; r3, v3 = state[8:10], state[10:12]
        r12 = np.linalg.norm(r2-r1); r13 = np.linalg.norm(r3-r1); r23 = np.linalg.norm(r3-r2)
        a1 = G*m2*(r2-r1)/r12**3 + G*m3*(r3-r1)/r13**3
        a2 = G*m1*(r1-r2)/r12**3 + G*m3*(r3-r2)/r23**3
        a3 = G*m1*(r1-r3)/r13**3 + G*m2*(r2-r3)/r23**3
        return np.concatenate([v1, a1, v2, a2, v3, a3])

    # 引数に質量を渡す
    y = odeint(model, state0, t, args=(m1, m2, m3))
    
    data = []
    for i in range(len(t)):
        data.append({"Time": t[i], "Body": "星1 (青)", "x": y[i,0], "y": y[i,1], "Size": m1})
        data.append({"Time": t[i], "Body": "星2 (赤)", "x": y[i,4], "y": y[i,5], "Size": m2})
        data.append({"Time": t[i], "Body": "星3 (緑)", "x": y[i,8], "y": y[i,9], "Size": m3})
    df_anim = pd.DataFrame(data)
    
    fig = px.scatter(
        df_anim, x="x", y="y", animation_frame="Time", animation_group="Body", 
        color="Body", size="Size", range_x=[-2, 2], range_y=[-2, 2]
    )
    # 軌跡を描画
    fig.add_trace(go.Scatter(x=y[:,0], y=y[:,1], mode='lines', line=dict(color='blue', width=1), opacity=0.3, showlegend=False))
    fig.add_trace(go.Scatter(x=y[:,4], y=y[:,5], mode='lines', line=dict(color='red', width=1), opacity=0.3, showlegend=False))
    fig.add_trace(go.Scatter(x=y[:,8], y=y[:,9], mode='lines', line=dict(color='green', width=1), opacity=0.3, showlegend=False))

    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 20
    
    latex = r"\vec{a}_i = \sum_{j \neq i} G m_j \frac{\vec{r}_j - \vec{r}_i}{|\vec{r}_j - \vec{r}_i|^3}"
    desc = """
    **三体問題 (The Three-Body Problem)**
    *   ニュートンの万有引力の法則です。
    *   物体が2つだけなら綺麗な楕円を描きますが、3つになった瞬間に**「一般解が存在しない（式では解けない）」**状態になります。
    *   質量を0.1変えるだけで、未来の軌道が予測不能に乱れる様子を確認してください。
    """
    return fig, desc, latex, None


# -------------------------------------------------------
# メイン処理
# -------------------------------------------------------
st.title(f"{field}")

# 選択肢に応じてモデル切り替え
df = None
y_range = None
fig_anim = None
latex_formula = ""

if "生物" in field:
    sub = st.sidebar.selectbox("モデル選択", ["生態系 (捕食)", "人口増加"])
    if "捕食" in sub: df, desc, latex_formula, y_range = run_lotka_volterra()
    else: df, desc, latex_formula, y_range = run_logistic()

elif "医療" in field:
    sub = st.sidebar.selectbox("モデル選択", ["感染症 (SIR)", "薬の効果 (血中濃度)"])
    if "感染" in sub: df, desc, latex_formula, y_range = run_sir()
    else: df, desc, latex_formula, y_range = run_drug()

elif "経済" in field:
    df, desc, latex_formula, y_range = run_bass()

elif "物理" in field:
    sub = st.sidebar.selectbox("モデル選択", ["バネの単振動", "冷却法則"])
    if "バネ" in sub: df, desc, latex_formula, y_range = run_spring()
    else: df, desc, latex_formula, y_range = run_cooling()

elif "恋愛" in field:
    df, desc, latex_formula, y_range = run_love()

elif "カオス" in field:
    fig_anim, desc, latex_formula, y_range = run_three_body()

# === 画面描画 ===

# 1. グラフ
if fig_anim:
    st.plotly_chart(fig_anim, use_container_width=True)
elif df is not None:
    y_cols = [c for c in df.columns if c != 'Time']
    fig = px.line(df, x='Time', y=y_cols)
    if y_range: fig.update_yaxes(range=y_range)
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

# 2. 数式と解説
st.markdown("---")
cols = st.columns([1, 1])

with cols[0]:
    st.subheader("📐 モデルの数式")
    st.latex(latex_formula)

with cols[1]:
    st.subheader("📝 解説")
    st.markdown(desc)