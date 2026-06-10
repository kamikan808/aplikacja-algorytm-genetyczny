import streamlit as st
import numpy as np
import ast
import time
import pandas as pd
import io
import plotly.graph_objects as go

from ga import (
    GAConfig, run_ga,
    himmelblau, sphere, rastrigin, rosenbrock,
    save_result # zostawione na wszelki
)

# config do stonki
st.set_page_config(page_title="Algorytm Genetyczny", layout="wide")
st.title("Optymalizacja funkcji - Algorytm Genetyczny")

# inicjalizacja historii dla pojedynczych uruchomień
if "history_rows" not in st.session_state:
    st.session_state.history_rows = []


DIMENSION_OPTIONS = [2, 5, 10, 20, 27]
PLOT_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "wykres_ga",
        "height": 900,
        "width": 1400,
        "scale": 3,
    },
}

FUNCTION_DEFS = {
    "Himmelblau": {
        "func": himmelblau,
        "range": (-5.0, 5.0),
        "fixed_n_vars": 2,
        "description": "Funkcja Himmelblau jest zdefiniowana tylko dla 2 zmiennych.",
    },
    "Sphere": {
        "func": sphere,
        "range": (-5.12, 5.12),
        "fixed_n_vars": None,
        "description": "Funkcja Sphere obsługuje dowolną liczbę zmiennych.",
    },
    "Rastrigin": {
        "func": rastrigin,
        "range": (-5.12, 5.12),
        "fixed_n_vars": None,
        "description": "Funkcja Rastrigin obsługuje dowolną liczbę zmiennych.",
    },
    "Rosenbrock": {
        "func": rosenbrock,
        "range": (-5.0, 10.0),
        "fixed_n_vars": None,
        "description": "Funkcja Rosenbrock obsługuje dowolną liczbę zmiennych.",
    },
}

ALLOWED_NUMPY_ATTRS = {
    "sin", "cos", "tan", "sqrt", "exp", "log", "log10", "abs",
    "sum", "mean", "min", "max", "power", "pi", "e",
}
ALLOWED_MATH_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "abs": np.abs,
    "sum": np.sum,
    "mean": np.mean,
}
ALLOWED_CONSTANTS = {"pi": np.pi, "e": np.e}
ALLOWED_NAMES = {"x", "np", *ALLOWED_MATH_FUNCS.keys(), *ALLOWED_CONSTANTS.keys()}
ALLOWED_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Subscript, ast.Slice, ast.Tuple, ast.List,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.UAdd, ast.USub, ast.Attribute,
)


def compile_custom_function(expression: str):
    """Zbuduj bezpieczną funkcję celu z wyrażenia wpisanego w GUI."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Błąd składni funkcji: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise ValueError("Funkcja zawiera niedozwolony element składni.")

        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
            raise ValueError(f"Niedozwolona nazwa w funkcji: {node.id}")

        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id != "np":
                raise ValueError("Dozwolone są tylko funkcje NumPy w postaci np.nazwa.")
            if node.attr.startswith("_") or node.attr not in ALLOWED_NUMPY_ATTRS:
                raise ValueError(f"Niedozwolona funkcja NumPy: np.{node.attr}")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in ALLOWED_MATH_FUNCS:
                    raise ValueError(f"Niedozwolone wywołanie funkcji: {node.func.id}()")
            elif isinstance(node.func, ast.Attribute):
                if not isinstance(node.func.value, ast.Name) or node.func.value.id != "np":
                    raise ValueError("Dozwolone są tylko wywołania np.nazwa(...).")
                if node.func.attr not in ALLOWED_NUMPY_ATTRS:
                    raise ValueError(f"Niedozwolone wywołanie funkcji: np.{node.func.attr}()")
            else:
                raise ValueError("Niedozwolone wywołanie funkcji.")

    compiled = compile(tree, "<custom_function>", "eval")

    def custom_func(x: np.ndarray) -> float:
        namespace = {
            "x": np.asarray(x, dtype=float),
            "np": np,
            **ALLOWED_MATH_FUNCS,
            **ALLOWED_CONSTANTS,
        }
        value = eval(compiled, {"__builtins__": {}}, namespace)
        value_array = np.asarray(value, dtype=float)
        if value_array.size == 1:
            return float(value_array)
        return float(np.sum(value_array))

    return custom_func


def evaluate_grid(function_name: str, func, x_values: np.ndarray, y_values: np.ndarray):
    X, Y = np.meshgrid(x_values, y_values)

    if function_name == "Himmelblau":
        Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2
    elif function_name == "Sphere":
        Z = X**2 + Y**2
    elif function_name == "Rastrigin":
        Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
    elif function_name == "Rosenbrock":
        Z = 100 * (Y - X**2)**2 + (1 - X)**2
    else:
        Z = np.vectorize(lambda a, b: func(np.array([a, b], dtype=float)))(X, Y)

    return X, Y, Z

# sidepanel
st.sidebar.header("Konfiguracja Algorytmu")

# Wybór funkcji
funkcja_nazwa = st.sidebar.selectbox(
    "Wybierz funkcję",
    list(FUNCTION_DEFS.keys()) + ["Własna funkcja"]
)

config_error = None
custom_expression = None

if funkcja_nazwa == "Własna funkcja":
    custom_expression = st.sidebar.text_area(
        "Wpisz własną funkcję f(x)",
        value="sum(x**2)",
        help="Używaj zmiennej x, np. x[0]**2 + x[1]**2, sum(x**2) albo sin(x**2) + cos(x**2). Jeśli wynik jest wektorem, aplikacja zsumuje jego elementy.",
        height=90,
    )
    try:
        wybrana_funkcja = compile_custom_function(custom_expression.strip())
    except ValueError as exc:
        wybrana_funkcja = None
        config_error = str(exc)

    liczba_zmiennych = int(st.sidebar.number_input(
        "Liczba zmiennych",
        min_value=1,
        max_value=50,
        value=2,
        step=1,
    ))
    lower = float(st.sidebar.number_input("Dolny zakres", value=-5.0, step=0.5, format="%.2f"))
    upper = float(st.sidebar.number_input("Górny zakres", value=5.0, step=0.5, format="%.2f"))
    if lower >= upper:
        config_error = "Dolny zakres musi być mniejszy od górnego."
else:
    selected_function = FUNCTION_DEFS[funkcja_nazwa]
    wybrana_funkcja = selected_function["func"]
    lower, upper = selected_function["range"]
    fixed_n_vars = selected_function["fixed_n_vars"]

    if fixed_n_vars is not None:
        liczba_zmiennych = int(st.sidebar.selectbox(
            "Liczba zmiennych",
            [fixed_n_vars],
            disabled=True,
        ))
        st.sidebar.info(selected_function["description"])
    else:
        liczba_zmiennych = st.sidebar.selectbox(
            "Liczba zmiennych",
            DIMENSION_OPTIONS,
            index=0,
        )
        st.sidebar.caption(selected_function["description"])

    st.sidebar.caption(f"Zakres poszukiwań: [{lower}, {upper}]")

if config_error:
    st.sidebar.error(config_error)

# populacja i epoki
pop_size = st.sidebar.number_input("Rozmiar populacji", min_value=10, max_value=1000, value=100, step=10)
epochs = st.sidebar.number_input("Liczba epok", min_value=10, max_value=5000, value=200, step=50)

# metody do ga
metoda_selekcji = st.sidebar.selectbox("Metoda selekcji", ["Najlepszych", "Ruletki", "Turniejowa"])
metoda_krzyzowania = st.sidebar.selectbox("Krzyżowanie", ["Jednopunktowe", "Dwupunktowe", "Jednorodne", "Ziarniste"])
metoda_mutacji = st.sidebar.selectbox("Mutacja", ["Brzegowa", "Jednopunktowa", "Dwupunktowa"])

# prawdopodobieństwa
p_crossover = st.sidebar.slider("Prawdopodobieństwo krzyżowania", 0.0, 1.0, 0.8)
p_mutation = st.sidebar.slider("Prawdopodobieństwo mutacji", 0.0, 1.0, 0.1)

typ_optymalizacji = st.sidebar.radio(
    "Typ optymalizacji",
    ["Minimalizacja", "Maksymalizacja"],
    horizontal=True,
)
maximize = typ_optymalizacji == "Maksymalizacja"
best_point_label = "Znalezione maksimum" if maximize else "Znalezione minimum"

inwersja = st.sidebar.checkbox("Zastosuj operator inwersji", value=False)
elitaryzm = st.sidebar.checkbox("Strategia elitarna", value=True)


# tabsy
tab1, tab2 = st.tabs(["Pojedyncze uruchomienie", "Pełny Benchmark Metod"])

sel_map = {"Najlepszych": "best", "Ruletki": "roulette", "Turniejowa": "tournament"}
cross_map = {"Jednopunktowe": "single_point", "Dwupunktowe": "two_point", "Jednorodne": "uniform", "Ziarniste": "granular"}
mut_map = {"Brzegowa": "boundary", "Jednopunktowa": "single_point", "Dwupunktowa": "two_point"}

# tab 1 - pojedyncze uruchomienie
with tab1:
    if st.button("Uruchom algorytm", type="primary", disabled=wybrana_funkcja is None or config_error is not None):
        
        cfg = GAConfig(
            func=wybrana_funkcja,
            n_vars=liczba_zmiennych,
            lower=lower,
            upper=upper,
            maximize=maximize,
            pop_size=pop_size,
            n_epochs=epochs,
            selection_method=sel_map[metoda_selekcji], 
            crossover_method=cross_map[metoda_krzyzowania],
            mutation_method=mut_map[metoda_mutacji],
            crossover_prob=p_crossover,
            mutation_prob=p_mutation,
            use_inversion=inwersja,
            use_elitism=elitaryzm
        )
        
        start_time = time.time()
        
        with st.spinner("Trwają obliczenia..."):
            try:
                r = run_ga(cfg)
            except Exception as exc:
                st.error(f"Nie udało się uruchomić algorytmu dla wybranej funkcji: {exc}")
                st.stop()
            
        end_time = time.time()
        czas_obliczen = end_time - start_time
        st.success("Obliczenia zakończone!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Czas obliczeń", f"{czas_obliczen:.4f} s")
        col2.metric("Najlepsza wartość f(x)", f"{r.best_value:.6f}")
        col3.metric("Liczba epok", epochs)
        
        st.write("**Najlepszy osobnik (X):**")
        st.code(r.best_individual)
        # wykres zbieżności
        st.subheader("📈 Wykres zbieżności algorytmu")
        
        epoki_x = list(range(1, len(r.history_best) + 1))
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=epoki_x,
            y=r.history_best,
            mode="lines",
            name="Najlepsza wartość",
            line=dict(color="#2563eb", width=3),
        ))
        fig_conv.add_trace(go.Scatter(
            x=epoki_x,
            y=r.history_avg,
            mode="lines",
            name="Średnia populacji",
            line=dict(color="#f59e0b", width=3, dash="dash"),
            yaxis="y2",
        ))
        fig_conv.update_layout(
            title=dict(text=f"Przebieg ewolucji dla funkcji {funkcja_nazwa}", x=0.02),
            template="plotly_white",
            height=480,
            margin=dict(l=40, r=60, t=70, b=45),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Epoka", showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
            yaxis=dict(title="Najlepsza wartość", showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
            yaxis2=dict(title="Średnia populacji", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(fig_conv, use_container_width=True, config=PLOT_CONFIG)
        nowy_wynik = {
            "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
            "func_name":        funkcja_nazwa,
            "n_vars":           cfg.n_vars,
            "maximize":         cfg.maximize,
            "lower":            cfg.lower,
            "upper":            cfg.upper,
            "pop_size":         cfg.pop_size,
            "n_epochs":         cfg.n_epochs,
            "precision":        cfg.precision,
            "selection_method": cfg.selection_method,
            "crossover_method": cfg.crossover_method,
            "mutation_method":  cfg.mutation_method,
            "crossover_prob":   cfg.crossover_prob,
            "mutation_prob":    cfg.mutation_prob,
            "use_inversion":    cfg.use_inversion,
            "inversion_prob":   cfg.inversion_prob,
            "use_elitism":      cfg.use_elitism,
            "n_elite":          cfg.n_elite,
            "best_value":       round(r.best_value, 10),
            "best_individual":  str([round(v, 6) for v in r.best_individual.tolist()]),
            "elapsed_time":     round(czas_obliczen, 4),
        }
        st.session_state.history_rows.append(nowy_wynik)
            # Wykresy 
        if liczba_zmiennych == 2:
            st.subheader("Wizualizacja wyników")
            
            x = np.linspace(lower, upper, 180)
            y = np.linspace(lower, upper, 180)
            X, Y, Z = evaluate_grid(funkcja_nazwa, wybrana_funkcja, x, y)
            
            best_x, best_y = r.best_individual[0], r.best_individual[1]
            best_z = r.best_value

            col_plot1, col_plot2 = st.columns(2)
            plot_height = 620
            
            with col_plot1:
                fig_3d = go.Figure(data=[go.Surface(z=Z, x=x, y=y, colorscale='Viridis', opacity=0.9)])
                fig_3d.add_trace(go.Scatter3d(
                    x=[best_x], y=[best_y], z=[best_z],
                    mode='markers',
                    marker=dict(color='cyan', size=6, symbol='diamond', line=dict(color='black', width=1)),
                    name=best_point_label
                ))
                
                fig_3d.update_layout(
                    title=f'Interaktywny model 3D - {funkcja_nazwa}',
                    template="plotly_white",
                    height=plot_height,
                    margin=dict(l=0, r=0, b=0, t=55),
                    scene=dict(
                        xaxis_title="x",
                        yaxis_title="y",
                        zaxis_title="f(x,y)",
                        aspectmode="cube",
                    ),
                )
                st.plotly_chart(fig_3d, use_container_width=True, config=PLOT_CONFIG)

            with col_plot2:
                fig_2d = go.Figure()
                fig_2d.add_trace(go.Contour(
                    x=x,
                    y=y,
                    z=Z,
                    colorscale="Inferno",
                    contours=dict(coloring="heatmap", showlines=False),
                    ncontours=60,
                    colorbar=dict(title="f(x,y)", thickness=16),
                    name="Wartość funkcji",
                ))
                fig_2d.add_trace(go.Scatter(
                    x=[best_x],
                    y=[best_y],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        color="#00e5ff",
                        size=18,
                        line=dict(color="black", width=1.5),
                    ),
                    name=best_point_label,
                ))
                fig_2d.update_layout(
                    title=f"Heatmapa - {funkcja_nazwa}",
                    template="plotly_white",
                    height=plot_height,
                    margin=dict(l=45, r=20, b=45, t=55),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(title="x", constrain="domain"),
                    yaxis=dict(title="y", scaleanchor="x", scaleratio=1),
                )
                st.plotly_chart(fig_2d, use_container_width=True, config=PLOT_CONFIG)


    # wyniki csv
    if st.session_state.history_rows:
        st.markdown("---")
        st.subheader("Historia uruchomień w tej sesji")
        df_history = pd.DataFrame(st.session_state.history_rows)
        st.dataframe(df_history, use_container_width=True)
        csv_buffer = io.StringIO()
        df_history.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        st.download_button(
            label="📥 Pobierz całą historię jako CSV (pojedyncze uruchomienia)",
            data=csv_data,
            file_name="wyniki_ga.csv",
            mime="text/csv"
        )


# tab 2 - ranking wszystkich kombinacji
with tab2:
    st.write("Ta opcja przetestuje automatycznie **wszystkie 36 kombinacji** (selekcja × krzyżowanie × mutacja) dla aktualnie wybranej funkcji i parametrów (populacja, epoki), a następnie wygeneruje ranking wyników.")
    
    if st.button(
        "Generuj Ranking (Może potrwać kilkadziesiąt sekund)",
        type="primary",
        key="btn_ranking",
        disabled=wybrana_funkcja is None or config_error is not None,
    ):
        
        lista_selekcji = list(sel_map.keys())
        lista_krzyzowan = list(cross_map.keys())
        lista_mutacji = list(mut_map.keys())
        
        total_runs = len(lista_selekcji) * len(lista_krzyzowan) * len(lista_mutacji)
        current_run = 0
        
        ranking_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_bench = time.time()
        
        for sel in lista_selekcji:
            for cross in lista_krzyzowan:
                for mut in lista_mutacji:
                    
                    status_text.text(f"Obliczanie: {sel} | {cross} | {mut} ({current_run+1}/{total_runs})")
                    
                    cfg_bench = GAConfig(
                        func=wybrana_funkcja,
                        n_vars=liczba_zmiennych,
                        lower=lower, upper=upper,
                        maximize=maximize,
                        pop_size=pop_size, n_epochs=epochs,
                        selection_method=sel_map[sel], 
                        crossover_method=cross_map[cross],
                        mutation_method=mut_map[mut],
                        crossover_prob=p_crossover, mutation_prob=p_mutation,
                        use_inversion=inwersja, use_elitism=elitaryzm
                    )
                    t0 = time.time()
                    try:
                        r_bench = run_ga(cfg_bench)
                    except Exception as exc:
                        st.error(f"Benchmark przerwany dla konfiguracji {sel} | {cross} | {mut}: {exc}")
                        st.stop()
                    t1 = time.time()
                    
                    ranking_data.append({
                        "Selekcja": sel,
                        "Krzyżowanie": cross,
                        "Mutacja": mut,
                        "Najlepsza wartość": round(r_bench.best_value, 8),
                        "Czas obliczeń [s]": round(t1 - t0, 4)
                    })
                    
                    current_run += 1
                    progress_bar.progress(current_run / total_runs)
        
        status_text.text(f"Zakończono! Całkowity czas benchmarku: {time.time() - start_bench:.2f} s")
        df_ranking = pd.DataFrame(ranking_data)
        df_ranking = df_ranking.sort_values(
            by="Najlepsza wartość",
            ascending=not maximize,
        ).reset_index(drop=True)
        
        df_ranking.index = df_ranking.index + 1
        df_ranking.index.name = "Pozycja"
        st.subheader("Wyniki Rankingu")
        st.dataframe(df_ranking, use_container_width=True)
        
        csv_buffer_rank = io.StringIO()
        df_ranking.to_csv(csv_buffer_rank)
        
        st.download_button(
            label="📥 Pobierz Ranking jako CSV (ranking_summary.csv)",
            data=csv_buffer_rank.getvalue(),
            file_name=f"ranking_summary_{funkcja_nazwa}_{liczba_zmiennych}D.csv",
            mime="text/csv",
            key="download_ranking"
        )
