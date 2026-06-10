import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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

# sidepanel
st.sidebar.header("Konfiguracja Algorytmu")

# Wybór funkcji
funkcja_nazwa = st.sidebar.selectbox(
    "Wybierz funkcję testową", 
    ["Himmelblau", "Sphere", "Rastrigin", "Rosenbrock"]
)

funkcje_map = {
    "Himmelblau": himmelblau,
    "Sphere": sphere,
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock
}
wybrana_funkcja = funkcje_map[funkcja_nazwa]

# liczba zmiennych
liczba_zmiennych = st.sidebar.selectbox("Liczba zmiennych", [2, 5, 10, 20, 27])

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

inwersja = st.sidebar.checkbox("Zastosuj operator inwersji", value=False)
elitaryzm = st.sidebar.checkbox("Strategia elitarna", value=True)


# tabsy
tab1, tab2 = st.tabs(["Pojedyncze uruchomienie", "Pełny Benchmark Metod"])

sel_map = {"Najlepszych": "best", "Ruletki": "roulette", "Turniejowa": "tournament"}
cross_map = {"Jednopunktowe": "single_point", "Dwupunktowe": "two_point", "Jednorodne": "uniform", "Ziarniste": "granular"}
mut_map = {"Brzegowa": "boundary", "Jednopunktowa": "single_point", "Dwupunktowa": "two_point"}

# tab 1 - pojedyncze uruchomienie
with tab1:
    if st.button("Uruchom algorytm", type="primary"):
        
        cfg = GAConfig(
            func=wybrana_funkcja,
            n_vars=liczba_zmiennych,
            lower=-5.0,
            upper=5.0,
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
            r = run_ga(cfg)
            
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
        
        fig_conv, ax_conv = plt.subplots(figsize=(12, 4), dpi=100)
        epoki_x = range(1, len(r.history_best) + 1)
        
        ax_conv.plot(epoki_x, r.history_best, label='Najlepsza wartość', color='blue', linewidth=2)
        ax_conv.plot(epoki_x, r.history_avg, label='Średnia populacji', color='orange', linestyle='--', linewidth=1.5)
        
        ax_conv.set_title(f"Przebieg ewolucji dla funkcji {funkcja_nazwa}")
        ax_conv.set_xlabel("Epoka")
        ax_conv.set_ylabel("Wartość funkcji f(x)")
        ax_conv.legend()
        ax_conv.grid(True, linestyle=':', alpha=0.7)
        
        st.pyplot(fig_conv)
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
            
            x = np.linspace(-5.0, 5.0, 200)
            y = np.linspace(-5.0, 5.0, 200)
            X, Y = np.meshgrid(x, y)
            
            if funkcja_nazwa == "Himmelblau":
                Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2
            elif funkcja_nazwa == "Sphere":
                Z = X**2 + Y**2
            elif funkcja_nazwa == "Rastrigin":
                Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
            elif funkcja_nazwa == "Rosenbrock":
                Z = 100 * (Y - X**2)**2 + (1 - X)**2
            
            best_x, best_y = r.best_individual[0], r.best_individual[1]
            best_z = r.best_value

            col_plot1, col_plot2 = st.columns(2)
            
            with col_plot1:
                fig_3d = go.Figure(data=[go.Surface(z=Z, x=x, y=y, colorscale='Viridis', opacity=0.9)])
                fig_3d.add_trace(go.Scatter3d(
                    x=[best_x], y=[best_y], z=[best_z],
                    mode='markers',
                    marker=dict(color='cyan', size=6, symbol='diamond', line=dict(color='black', width=1)),
                    name='Znalezione minimum'
                ))
                
                fig_3d.update_layout(
                    title=f'Interaktywny model 3D - {funkcja_nazwa}',
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                st.plotly_chart(fig_3d, use_container_width=True)

            with col_plot2:
                fig_2d, ax_2d = plt.subplots(figsize=(7, 6), dpi=100)
                heatmap = ax_2d.contourf(X, Y, Z, levels=50, cmap=cm.inferno)
                ax_2d.set_title(f'Heatmapa - {funkcja_nazwa}')
                fig_2d.colorbar(heatmap, ax=ax_2d)
                
                ax_2d.plot(best_x, best_y, marker='*', color='cyan', markersize=15, markeredgecolor='black', label='Znalezione minimum')
                ax_2d.legend()
                
                st.pyplot(fig_2d)


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
    
    if st.button("Generuj Ranking (Może potrwać kilkadziesiąt sekund)", type="primary", key="btn_ranking"):
        
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
                        lower=-5.0, upper=5.0,
                        pop_size=pop_size, n_epochs=epochs,
                        selection_method=sel_map[sel], 
                        crossover_method=cross_map[cross],
                        mutation_method=mut_map[mut],
                        crossover_prob=p_crossover, mutation_prob=p_mutation,
                        use_inversion=inwersja, use_elitism=elitaryzm
                    )
                    t0 = time.time()
                    r_bench = run_ga(cfg_bench)
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
        df_ranking = df_ranking.sort_values(by="Najlepsza wartość", ascending=True).reset_index(drop=True)
        
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
