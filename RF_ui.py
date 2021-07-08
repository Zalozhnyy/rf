import os
import sys

sys.path.append(os.path.dirname(__file__))

from typing import List
from dataclasses import dataclass
import locale

import tkinter as tk
from tkinter import ttk

from tkinter import simpledialog as sd
from tkinter import filedialog as fd
from tkinter import messagebox as mb

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

matplotlib.use('TkAgg')

import numpy as np

from polus_fit_v0 import main

disp_header = '''#Dispersion model layer index
<Layer>
{layer_index:d}
#Model type (1-Plushenkov, 2-Debay, 3-Drude)
<Type>
1
#Number of poles
<NPoles>
{poles_number:d}
#Poles Ak Ck format
<Poles>'''


class DataParser:
    def __init__(self, path):
        self.path = path
        if path:
            self.dir_path = os.path.dirname(self.path)
        self.decoding_def = locale.getpreferredencoding()
        self.decoding = 'utf-8'

    def lay_decoder(self, lines: list = None) -> (np.ndarray, dict):
        #### .LAY DECODER
        if not lines:
            try:
                with open(rf'{self.path}', 'r', encoding=f'{self.decoding}') as file:
                    lines = file.readlines()
            except UnicodeDecodeError:

                with open(rf'{self.path}', 'r', encoding=f'{self.decoding_def}') as file:
                    lines = file.readlines()

        try:

            line = 2  # <Количество слоев> + 1 строка
            lay_numeric = int(lines[line])
            out_lay = np.zeros((lay_numeric, 3), dtype=int)
            output_dict = {}
            # print(f'<Количество слоев> + 1 строка     {lines[line]}')

            line += 2  # <Номер, название слоя>
            # print(f'<Номер, название слоя>     {lines[line]}')

            for layer in range(lay_numeric):
                line += 1  # <Номер, название слоя> + 1 строка
                number, name = lines[line].strip().split()
                out_lay[layer, 0] = int(lines[line].split()[0])  # 0 - номер слоя

                line += 2  # <газ(0)/не газ(1), и тд + 1 строка
                out_lay[layer, 1] = int(lines[line].split()[2])  # 1 - стороннй ток
                out_lay[layer, 2] = int(lines[line].split()[3])  # 2 - стро. ист.
                output_dict.update({int(number): {'name': name,
                                                  'ppn': int(lines[line].split()[0])}})

                extended = False
                if int(lines[line].split()[-1]) == 1:
                    extended = True

                line += 2  # <давление в слое(атм.), плотн.(г/см3), + 1 строка
                output_dict[int(number)].update({'material_N': int(lines[line].strip().split()[0])})

                if extended is False:
                    line += 2  # следущая частица    <Номер, название слоя>
                elif extended is True:
                    line += 2  # <молекулярный вес[г/моль] + 1 строка

                    line += 2  # следущая частица    <Номер, название слоя>
            return out_lay, output_dict
        except Exception:
            print('Ошибка в чтении файла .LAY')
            return None, None


@dataclass
class Data:
    number_poles: int
    init_data: List[float]

    tau: float
    epsilon: float

    experimental_data: str = None
    layer: int = -1
    ak: np.ndarray = np.empty(0)
    ck: np.ndarray = np.empty(0)


@dataclass
class Solutions:
    poles: int
    init1: float
    init2: float

    tau: float
    epsilon: float

    x: np.ndarray = None
    y: np.ndarray = None
    yn: np.ndarray = None
    yst: np.ndarray = None
    yi: np.ndarray = None
    yin: np.ndarray = None
    ak: np.ndarray = None
    ck: np.ndarray = None

    id: int = -1


class ExtendedExperimentFrame(ttk.LabelFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent)
        self._parent = parent
        self._main_figure, self._main_ax, self._canvas = kwargs['fig']
        self._data: Data = kwargs['data']

        self['text'] = 'Расширенные начальные параметры'

        self._N_value: tk.StringVar = tk.StringVar(value=str(1))
        self._sigma_dc_value: tk.StringVar = tk.StringVar(value=f"{1:.4E}")

        self._alpha_value: List[tk.StringVar] = [tk.StringVar(value=f'{1:.3E}') for _ in range(3)]
        self._beta_value: List[tk.StringVar] = [tk.StringVar(value=f'{1:.3E}') for _ in range(3)]
        self._delta_epsilon_value: List[tk.StringVar] = [tk.StringVar(value=f'{1:.3E}') for _ in range(3)]
        self._tau_nh_value: List[tk.StringVar] = [tk.StringVar(value=f'{8e-8:.3E}') for _ in range(3)]
        self._epsilon_inf_value: List[tk.StringVar] = [tk.StringVar(value=str(12)) for _ in range(3)]

        self._N_entry: tk.Entry
        self._sigma_dc_entry: tk.Entry

        self._alpha_entry: List[tk.Entry] = []
        self._beta_entry: List[tk.Entry] = []
        self._delta_epsilon_entry: List[tk.Entry] = []
        self._tau_nh_entry: List[tk.Entry] = []
        self._epsilon_inf_entry: List[tk.Entry] = []

        self.start_calculate: tk.Button

        self._init_formula()
        self._init_entry()

    def change_sigma_dc_values(self):
        data = np.loadtxt(self._data.experimental_data, dtype=float)
        x, y, yi = data[:, 0], data[:, 1], data[:, 2]
        sigma_dc = y[np.argmin(x)]
        self._sigma_dc_value.set(f"{sigma_dc:.4E}")

    def _init_entry(self):
        row = 2

        N_label = tk.Label(self, text='N:')
        N_label.grid(row=row, column=0, columnspan=1, pady=2)

        self._poles_entry = tk.Entry(self, textvariable=self._N_value, width=10)
        self._poles_entry.grid(row=row, column=1, columnspan=1, padx=5, pady=2)

        row += 1

        k_label = tk.Label(self, text='k')
        k_label.grid(row=row, column=0, columnspan=1, pady=2)
        k_entry = []
        k_entry_value = [tk.StringVar(value=str(i + 1)) for i in range(3)]
        for i in range(3):
            k_entry.append(tk.Entry(self, textvariable=k_entry_value[i], width=6, state='disabled', justify='center'))
            k_entry[i].grid(row=row, column=1 + i, columnspan=1, padx=5, pady=2)

        row += 1

        alpha_label = tk.Label(self, text='alpha:')
        alpha_label.grid(row=row, column=0, columnspan=1, pady=2)

        for i in range(3):
            self._alpha_entry.append(tk.Entry(self, textvariable=self._alpha_value[i], width=10))
            self._alpha_entry[i].grid(row=row, column=1 + i, columnspan=1, padx=5, pady=2)

        row += 1

        alpha_label = tk.Label(self, text='beta:')
        alpha_label.grid(row=row, column=0, columnspan=1, pady=2)

        for i in range(3):
            self._beta_entry.append(tk.Entry(self, textvariable=self._beta_value[i], width=10))
            self._beta_entry[i].grid(row=row, column=1 + i, columnspan=1, padx=5, pady=2)

        row += 1

        _delta_epsilon_entry_label = tk.Label(self, text='Δε:')
        _delta_epsilon_entry_label.grid(row=row, column=0, columnspan=1, pady=2)

        for i in range(3):
            self._delta_epsilon_entry.append(tk.Entry(self, textvariable=self._delta_epsilon_value[i], width=10))
            self._delta_epsilon_entry[i].grid(row=row, column=1 + i, columnspan=1, padx=5, pady=2)

        row += 1

        _delta_epsilon_entry_label = tk.Label(self, text='tau_nh:')
        _delta_epsilon_entry_label.grid(row=row, column=0, columnspan=1, pady=2)

        for i in range(3):
            self._tau_nh_entry.append(tk.Entry(self, textvariable=self._tau_nh_value[i], width=10))
            self._tau_nh_entry[i].grid(row=row, column=1 + i, columnspan=1, padx=5, pady=2)

        row += 1

        _epsilon_inf_label = tk.Label(self, text='ε infinity k:')
        _epsilon_inf_label.grid(row=row, column=0, columnspan=1, pady=2)

        for i in range(3):
            self._epsilon_inf_entry.append(tk.Entry(self, textvariable=self._epsilon_inf_value[i], width=10))
            self._epsilon_inf_entry[i].grid(row=row, column=1 + i, columnspan=1, padx=5, pady=2)

        row += 1

        _sigma_dc_label = tk.Label(self, text='σ_dc:')
        _sigma_dc_label.grid(row=row, column=0, columnspan=1, pady=2)

        self._sigma_dc_entry = tk.Entry(self, textvariable=self._sigma_dc_value, width=10)
        self._sigma_dc_entry.grid(row=row, column=1, columnspan=1, padx=5, pady=2)

        self.start_calculate = tk.Button(self, text='Отрисовать',
                                         command=self._add_graphic,
                                         width=16, state='disabled')
        self.start_calculate.grid(row=row + 1, column=0, columnspan=2, padx=5, pady=2)

    def _add_graphic(self):
        if not self._data.experimental_data:
            mb.showerror('ошибка', 'Файл эксперимнта ещё не выбран')
            return
        data = np.loadtxt(self._data.experimental_data, dtype=float)
        x, y, yi = data[:, 0], data[:, 1], data[:, 2]

        val = self._validation()
        if not val:
            return

        Y_base = yi / (x * 2 * np.pi * 8.85e-12 * 9e9)

        sigma_dc = val['sigma_dc'] / 9e9
        epsilon_0 = 8.85e-12

        Y = np.zeros_like(x)
        Yi = np.zeros_like(x)

        for i in range(Y.shape[0]):
            Y[i] += -(-1) * (sigma_dc / (epsilon_0 * x[i] * np.pi * 2)) ** val['N']

        for i in range(Yi.shape[0]):
            for j in range(3):
                Yi[i] += val['delta_epsilon'][j] / (
                        1 + (-1 * x[i] * np.pi * 2 * val['tau_nh'][j]) ** val['alpha'][j]) ** \
                         val['beta'][j] + val['eps_inf'][j]

        self._main_figure.clf()
        self._main_ax = self._main_figure.add_subplot(111)

        self._main_ax.semilogx(x, Y, label='ε"ex мнимая часть диэлектрической пронициаемости')
        self._main_ax.semilogx(x, Yi, label="ε'ex действительная часть диэлектрической пронициаемости")

        self._main_ax.semilogx(x, Y_base, 'r-.', label='ε" мнимая часть диэлектрической пронициаемости', color='red')
        self._main_ax.semilogx(x, y, 'r-.', label="ε' действительная часть диэлектрической пронициаемости",
                               color='blue')
        self._main_ax.set_xlabel("Гц")
        self._main_ax.grid(True)
        self._main_ax.set_title('Результаты эксперимента по измерению диэлектрической проницаемости')
        self._main_ax.legend(loc='best', fancybox=True, shadow=False)

        self._main_figure.tight_layout()

        self._canvas.draw()

    def _validation(self):

        alpha = self._try_ex(self._alpha_value)
        beta = self._try_ex(self._beta_value)
        delta_epsilon = self._try_ex(self._delta_epsilon_value)
        tau_nh = self._try_ex(self._tau_nh_value)
        eps_inf = self._try_ex(self._epsilon_inf_value)

        out = {'N': self._try_ex([self._N_value])[0],
               'sigma_dc': self._try_ex([self._sigma_dc_value])[0],
               'alpha': alpha,
               'beta': beta,
               'delta_epsilon': delta_epsilon,
               'tau_nh': tau_nh,
               'eps_inf': eps_inf,
               }

        for key, item in out.items():
            if not item:
                mb.showerror('ошибка', f'Обнаружено нечитаемое значение в {key}')
                return None
        return out

    def _try_ex(self, stringvar_list: list):
        out = []
        for i in stringvar_list:
            try:
                tmp = float(i.get())
            except ValueError:
                return None

            out.append(tmp)
        return out

    def _init_formula(self):
        f_frame = tk.Frame(self)
        f_frame.grid(row=0, column=0, columnspan=7, rowspan=2)

        label = tk.Label(self)
        label.grid(row=0, column=0, columnspan=7, rowspan=2)

        fig = plt.Figure(figsize=(5, 0.6), dpi=100)
        ax = fig.add_subplot(111)

        canvas = FigureCanvasTkAgg(fig, master=label)
        canvas.get_tk_widget().pack(side='top', fill='both', expand=0)

        ax.axis('off')
        tmptext = r"$\epsilon *(\omega) = " \
                  r"\epsilon^\prime" \
                  r"- i\epsilon^{\prime\prime} = " \
                  r"-i\left( \frac{\sigma_{dc}}{\epsilon_0 \omega} \right)^N +" \
                  r"\sum_{k=1}^3 \left[ {" \
                  r"\frac" \
                  r"{\Delta\epsilon_k}" \
                  r"{(1 + (i\omega\tau_{HN})^{\alpha_k})^{\beta_k} } + \epsilon_{\infty k}" \
                  r"} \right] $"

        ax.text(0, 0.7, tmptext, fontsize=13)
        canvas.draw()

        fig.tight_layout()


class MainWindow(tk.Frame):
    def __init__(self, parent, path):
        super().__init__(parent)

        self._parent: tk.Tk = parent
        self._path: str = path

        self._layers_numbers = self._init_lay_file()

        self._solutions: List[Solutions] = []

        js = {'number_poles': 9, "init_data": [0, 0], 'tau': 1e-8, 'epsilon': 1.0}

        self._data = Data(**js)

        self._mw_frame: ttk.Frame
        self._extended_frame: ExtendedExperimentFrame

        self._file_name_label: tk.Label
        self._poles_entry: tk.Entry
        self._poles_entry_value: tk.StringVar
        self._show_graph: tk.Button
        self._save: tk.Button
        self._dispersion_file_name_button: tk.Button

        self._select_solution_combobox: ttk.Combobox
        self._select_layer_combobox: ttk.Combobox

        self._notebook: ttk.Notebook
        self._notebook_frame: ttk.LabelFrame

        self._nb_list1_frame: ttk.Frame
        self._nb_list2_frame: ttk.LabelFrame
        self._nb_list3_frame: ttk.LabelFrame

        self._canvas1: FigureCanvasTkAgg
        self._canvas2: FigureCanvasTkAgg
        self._canvas3: FigureCanvasTkAgg

        self._figure1: plt.Figure
        self._figure2: plt.Figure
        self._figure3: plt.Figure

        self._ax1: plt.Axes
        self._ax2: plt.Axes
        self._ax3: plt.Axes

        self.rb1_x: tk.Checkbutton
        self.rb1_y: tk.Checkbutton
        self.rb2_x: tk.Checkbutton
        self.rb2_y: tk.Checkbutton
        self.rb3_x: tk.Checkbutton
        self.rb3_y: tk.Checkbutton

        self.rb1_x_value: tk.BooleanVar
        self.rb1_y_value: tk.BooleanVar
        self.rb2_x_value: tk.BooleanVar
        self.rb2_y_value: tk.BooleanVar
        self.rb3_x_value: tk.BooleanVar
        self.rb3_y_value: tk.BooleanVar

        self._solutions_ids: List[int] = []

        self._init_ui()

        parent.protocol("WM_DELETE_WINDOW", self.__onExit)

        self.grid(padx=10)

    def __onExit(self):
        self._parent.destroy()
        self.quit()
        self._parent.quit()

    def _init_lay_file(self) -> List[int]:
        f = None
        for file in os.listdir(self._path):
            if file.endswith('.LAY') or file.endswith('.lay'):
                f = file
                break
        if not f:
            return []

        dp = DataParser(os.path.join(self._path, f))
        _, lay_dict = dp.lay_decoder()
        if not lay_dict:
            return []

        out = []
        for key in lay_dict.keys():
            if int(lay_dict[key]['material_N']) == 2:
                out.append(key)

        return out

    def _init_ui(self):
        row = 0

        self._mw_frame = ttk.Frame(self)
        self._mw_frame.grid(row=0, column=0, sticky='NWSE')

        self._file_name_label = tk.Label(self._mw_frame, text='Файл не выбран')
        self._file_name_label.grid(row=row, column=0, columnspan=4)

        self._dispersion_file_name_button = tk.Button(self._mw_frame, text='Выбор файла эксперимента',
                                                      command=lambda: self._choice_file(False))
        self._dispersion_file_name_button.grid(row=row, column=4, columnspan=3, padx=10, sticky='W')

        row += 1

        self._init_layers_interface(row)

        row += 1

        poles_label = tk.Label(self._mw_frame, text='Количество полюсов :')
        poles_label.grid(row=row, column=0, columnspan=2, pady=5)

        self._poles_entry_value = tk.StringVar(value=str(self._data.number_poles))
        self._poles_entry_value.trace('w', self._change_poles_callback)

        self._poles_entry = tk.Entry(self._mw_frame, textvariable=self._poles_entry_value, width=10)
        self._poles_entry.grid(row=row, column=2, columnspan=5, padx=10, pady=5)

        row += 1

        init_label = tk.Label(self._mw_frame, text='Начальные значения (tau_k, Δε_k) :')
        init_label.grid(row=row, column=0, columnspan=2, pady=5)

        self._init_entry_tau_k = tk.StringVar(value=f'{self._data.init_data[0]:.4E}')
        self._init_entry_epsilon = tk.StringVar(value=f'{self._data.init_data[1]:.4E}')

        self._init_entry1 = tk.Entry(self._mw_frame, textvariable=self._init_entry_tau_k, width=15)
        self._init_entry1.grid(row=row, column=2, columnspan=2, padx=5, pady=5)

        self._init_entry2 = tk.Entry(self._mw_frame, textvariable=self._init_entry_epsilon, width=15)
        self._init_entry2.grid(row=row, column=4, columnspan=2, padx=5, pady=5)

        self._init_entry_tau_k.trace('w', self._change_tau_k_callback)
        self._init_entry_epsilon.trace('w', self._change_epsilon_callback)

        row += 1
        self._init_notebook()

        self._init_extended_experiment_ui(row)
        row += 40

        self._show_graph = tk.Button(self._mw_frame, text='Создать апроксимацию',
                                     command=self._on_show_plot_clicked,
                                     width=20,
                                     state='disabled')
        self._show_graph.grid(row=row, column=0, columnspan=3, padx=10, pady=10, sticky='W')

        self._clear_graph = tk.Button(self._mw_frame, text='Очистить графики и решения',
                                      command=self._clear_graph_and_solutions,
                                      width=25,
                                      state='disabled')
        self._clear_graph.grid(row=row, column=3, columnspan=5, padx=10, pady=10, sticky='W')
        row += 1

        self._save = tk.Button(self._mw_frame, text='Сохранить решение №', command=self._save_disp, state='disabled',
                               width=20)
        self._save.grid(row=row, column=0, columnspan=2, padx=10, pady=10, sticky='W')

        self._select_solution_combobox = ttk.Combobox(self._mw_frame, value=[val for val in self._solutions_ids],
                                                      width=5, state='readonly')
        self._select_solution_combobox.grid(row=row, column=2, columnspan=2, padx=2, pady=10, sticky='WN')

        self.bind_class(self._select_solution_combobox, "<<ComboboxSelected>>", self.__on_select_combobox_value)

        init_tau = self._data.tau
        init_epsilon = self._data.epsilon

        self._init_entry_tau_k.set(str(init_tau))
        self._init_entry_epsilon.set(str(init_epsilon))

        self.update()
        w = self._mw_frame.winfo_width() + 50
        h = self._mw_frame.winfo_height() + 50
        self._parent.geometry(f'{w}x{h}')

    def _init_layers_interface(self, row):
        layer_label = tk.Label(self._mw_frame, text='Введите слой :')
        layer_label.grid(row=row, column=0, columnspan=2, pady=5)

        self._select_layer_combobox = ttk.Combobox(self._mw_frame, value=[val for val in self._layers_numbers],
                                                   width=5, state='readonly')
        self._select_layer_combobox.grid(row=row, column=2, columnspan=2, padx=2, pady=10, sticky='WN')
        self.bind_class(self._select_layer_combobox, "<<ComboboxSelected>>", self._layer_combobox_selected_callback)

        if not self._layers_numbers:
            # mb.showerror('Ошибка', 'Отсутствуют диэлектические слои. Сохранение будет невозможно')
            pass
        else:
            self._select_layer_combobox.set(str(self._layers_numbers[0]))
            self._data.layer = self._layers_numbers[0]

    def _init_notebook(self):
        self._notebook_frame = ttk.LabelFrame(self._mw_frame, text='Графики')
        self._notebook_frame.grid(row=0, column=7, rowspan=60, columnspan=60, sticky='NWSE')

        self._notebook = ttk.Notebook(self._notebook_frame)
        self._notebook.grid(sticky='NWSE')

        self._nb_list1_frame = tk.Frame(self._notebook)
        self._nb_list2_frame = tk.Frame(self._notebook)
        self._nb_list3_frame = tk.Frame(self._notebook)

        self._notebook.add(self._nb_list1_frame, text=f'Эксперимент')
        self._notebook.add(self._nb_list2_frame, text=f'Диэлектрическая проницаемость')
        self._notebook.add(self._nb_list3_frame, text=f'Электропроводность')

        self._init_canvas()

        self.rb1_x_value = tk.BooleanVar(self._nb_list1_frame, value=True)
        self.rb1_y_value = tk.BooleanVar(self._nb_list1_frame, value=False)
        self.rb2_x_value = tk.BooleanVar(self._nb_list2_frame, value=True)
        self.rb2_y_value = tk.BooleanVar(self._nb_list2_frame, value=False)
        self.rb3_x_value = tk.BooleanVar(self._nb_list3_frame, value=True)
        self.rb3_y_value = tk.BooleanVar(self._nb_list3_frame, value=True)

        self.rb1_x = tk.Checkbutton(self._nb_list1_frame, text='Логарифмическая абциса',
                                    variable=self.rb1_x_value, state='disabled',
                                    command=lambda: self._change_x_log(self._ax1, self.rb1_x_value.get(),
                                                                       self._canvas1))
        self.rb1_y = tk.Checkbutton(self._nb_list1_frame, text='Логарифмическая ордината',
                                    variable=self.rb1_y_value, state='disabled',
                                    command=lambda: self._change_y_log(self._ax1, self.rb1_y_value.get(),
                                                                       self._canvas1))

        self.rb2_x = tk.Checkbutton(self._nb_list2_frame, text='Логарифмическая абциса',
                                    variable=self.rb2_x_value, state='disabled',
                                    command=lambda: self._change_x_log(self._ax2, self.rb2_y_value.get(),
                                                                       self._canvas2))
        self.rb2_y = tk.Checkbutton(self._nb_list2_frame, text='Логарифмическая ордината',
                                    variable=self.rb2_y_value, state='disabled',
                                    command=lambda: self._change_y_log(self._ax2, self.rb2_y_value.get(),
                                                                       self._canvas2))

        self.rb3_x = tk.Checkbutton(self._nb_list3_frame, text='Логарифмическая абциса',
                                    variable=self.rb3_x_value, state='disabled',
                                    command=lambda: self._change_x_log(self._ax3, self.rb3_x_value.get(),
                                                                       self._canvas3))
        self.rb3_y = tk.Checkbutton(self._nb_list3_frame, text='Логарифмическая ордината',
                                    variable=self.rb3_y_value, state='disabled',
                                    command=lambda: self._change_y_log(self._ax3, self.rb3_y_value.get(),
                                                                       self._canvas3))

        self.rb1_x.pack(side='left')
        self.rb1_y.pack(side='left')
        self.rb2_x.pack(side='left')
        self.rb2_y.pack(side='left')
        self.rb3_x.pack(side='left')
        self.rb3_y.pack(side='left')

    def _init_canvas(self):
        self._figure1 = plt.Figure(figsize=(12, 8), dpi=100)
        self._figure2 = plt.Figure(figsize=(12, 8), dpi=100)
        self._figure3 = plt.Figure(figsize=(12, 8), dpi=100)

        self._ax1 = self._figure1.add_subplot(111)
        self._ax2 = self._figure2.add_subplot(111)
        self._ax3 = self._figure3.add_subplot(111)

        self._canvas1 = FigureCanvasTkAgg(self._figure1, master=self._nb_list1_frame)
        self._canvas2 = FigureCanvasTkAgg(self._figure2, master=self._nb_list2_frame)
        self._canvas3 = FigureCanvasTkAgg(self._figure3, master=self._nb_list3_frame)

        toolbar1 = NavigationToolbar2Tk(self._canvas1, self._nb_list1_frame, pack_toolbar=True)
        toolbar2 = NavigationToolbar2Tk(self._canvas2, self._nb_list2_frame, pack_toolbar=True)
        toolbar3 = NavigationToolbar2Tk(self._canvas3, self._nb_list3_frame, pack_toolbar=True)

        toolbar1.update()
        toolbar2.update()
        toolbar3.update()

        self._canvas1.draw()
        self._canvas2.draw()
        self._canvas3.draw()

        self._canvas1.get_tk_widget().pack(side='top', fill='both', expand=True)
        self._canvas2.get_tk_widget().pack(side='top', fill='both', expand=True)
        self._canvas3.get_tk_widget().pack(side='top', fill='both', expand=True)

    def _init_extended_experiment_ui(self, row: int):

        self._extended_frame = ExtendedExperimentFrame(self._mw_frame,
                                                       **{'fig': [self._figure1, self._ax1, self._canvas1],
                                                          'data': self._data})
        self._extended_frame.grid(row=row, column=0, rowspan=36, columnspan=7, sticky='NWSE', padx=5, pady=5)

    # def _change_layer_callback(self, *args):
    #     tmp = self._select_layer_combobox.get()
    #     if tmp == '':
    #         return
    #     if all([i.isdigit() for i in tmp]):
    #         self._data.layer = int(tmp)
    #         self._select_layer_combobox['bg'] = '#FFFFFF'
    #         self._save_button_control(True)
    #
    #     else:
    #         self._select_layer_combobox['bg'] = '#F08080'
    #         self._save_button_control(False)

    def _layer_combobox_selected_callback(self, event):
        self._data.layer = int(self._select_layer_combobox.get())

    def _change_poles_callback(self, *args):
        tmp = self._poles_entry_value.get()
        if tmp == '':
            return
        try:
            tmp = int(tmp)
            if int(tmp) < 2:
                raise Exception

            self._data.number_poles = int(tmp)
            self._poles_entry['bg'] = '#FFFFFF'
            self._save_button_control(True)

        except Exception:
            self._poles_entry['bg'] = '#F08080'
            self._save_button_control(False)

    def _change_tau_k_callback(self, *args):
        tau_k = self._init_entry_tau_k.get()
        if tau_k == '':
            return
        try:
            tau_k = float(tau_k)
            self._init_entry1['bg'] = '#FFFFFF'
            self._init_entry2['bg'] = '#FFFFFF'

        except Exception:
            self._init_entry1['bg'] = '#F08080'
            self._init_entry2['bg'] = '#F08080'
            self._show_graph['state'] = 'disabled'
            return

        try:
            Ak = -1 / tau_k
        except ZeroDivisionError:
            return
        self._data.init_data[0] = Ak
        self._data.tau = tau_k
        self._change_epsilon_callback()

    def _change_epsilon_callback(self, *args):
        tau_k = self._init_entry_tau_k.get()
        epsilon = self._init_entry_epsilon.get()

        try:
            tau_k = float(tau_k)
        except Exception:
            self._init_entry1['bg'] = '#F08080'
            self._show_graph['state'] = 'disabled'
            return
        try:
            epsilon = float(epsilon)
            self._init_entry2['bg'] = '#FFFFFF'
        except Exception:
            self._init_entry2['bg'] = '#F08080'
            self._show_graph['state'] = 'disabled'
            return

        Ck = epsilon / tau_k
        self._data.init_data[1] = Ck
        self._data.tau = tau_k
        self._data.epsilon = epsilon
        if self._data.experimental_data:
            self._show_graph['state'] = 'normal'
        # print(f'{self._data.init_data[0]:.5E}   {self._data.init_data[1]:.5E}')

    def _choice_file(self, delete_calc: bool, debug: str = None):
        if not debug:
            f = fd.askopenfilename(title=f'Выберите файл дисперсии')
            if f == '':
                return
        else:
            f = debug

        with open(f, 'r') as ff:
            first_line = ff.readline().strip().split()

        if len(first_line) != 3:
            print(
                f'Файл эксперимента содержит 3 колонки первая колонка — частота Гц, вторая — эпсилон, третья — проводимость 1/с.\n'
                f'количество колонок выбранного файла {len(first_line)}')
            return

        self._path = self._path if self._path else os.path.dirname(f)
        self._data.experimental_data = f
        self._file_name_label['text'] = os.path.basename(f)
        self._extended_frame.change_sigma_dc_values()
        self._extended_frame.start_calculate['state'] = 'normal'
        self._activate_log_graph_changer()

        if len(self._solutions) > 0 or delete_calc:
            # mb.showinfo('Изменен файл эксперимента', 'Изменён файл эксперимента. Прошлые расчеты удалены.')
            print('Изменён файл эксперимента. Прошлые расчеты удалены.')
            self._change_file_deconstructor()

            self._ax1 = self._figure1.add_subplot(111)
            self._ax2 = self._figure2.add_subplot(111)
            self._ax3 = self._figure3.add_subplot(111)

        self._save_button_control(True)
        self._init_experimental_data_graphic()
        self._dispersion_file_name_button['command'] = lambda: self._choice_file(True)

    def _save_button_control(self, activate: bool):
        if self._data.experimental_data and os.path.exists(self._data.experimental_data):
            self._show_graph['state'] = 'normal'
        else:
            self._save['state'] = 'disabled'
            self._show_graph['state'] = 'disabled'
            return

        if self._data.layer == -1:
            self._save['state'] = 'disabled'
            return

        if self._data.ck.shape[0] > 0:
            self._save['state'] = 'normal'
        else:
            return

        if activate:
            self._save['state'] = 'normal'
        else:
            self._save['state'] = 'disabled'

    def _on_show_plot_clicked(self):
        db = {
            'file': self._data.experimental_data,
            'number_poles': self._data.number_poles,
            'init_data': self._data.init_data,
        }

        s = Solutions(*[self._data.number_poles, self._data.init_data[0], self._data.init_data[1], self._data.tau,
                        self._data.epsilon])
        s.id = len(self._solutions) + 1

        same = self._is_calculated(s)
        if same:
            print(f'Такое решение уже было рассчитано под номером {same}')
            mb.showinfo('info', f'Такое решение уже было рассчитано под номером {same}')
            return

        calc_results = dict(zip(['x', 'y', 'yn', 'yst', 'yi', 'yin', 'ak', 'ck'], main(db)))

        if any([i is None for i in calc_results.values()]):
            mb.showerror('Ошибка', f'Решение не сходится при Am={self._data.init_data[0]} Cm={self._data.init_data[1]}')
            return

        self._data.ak = calc_results['ak']
        self._data.ck = calc_results['ck']
        s.x, s.y, s.yn, s.yst, s.yi, s.yin, s.ak, s.ck = calc_results.values()

        self._solutions.append(s)
        self._add_solution_id_to_interface(s.id)

        self._draw_plot_to_canvas(s)

        self._clear_graph['state'] = 'normal'
        self._save_button_control(True)
        if self._notebook.index(self._notebook.select()) == 0:
            self._notebook.select(1)

    def _add_solution_id_to_interface(self, id: int):
        self._solutions_ids.append(id)
        old_value = self._select_solution_combobox.get()
        self._select_solution_combobox['value'] = [value for value in self._solutions_ids]

        if old_value != '':
            self._select_solution_combobox.set(old_value)
        else:
            self._select_solution_combobox.set(self._solutions_ids[0])

    def _is_calculated(self, s: Solutions) -> int:
        for i in range(len(self._solutions)):
            if s.poles == self._solutions[i].poles and s.init1 == self._solutions[i].init1 and s.init2 == \
                    self._solutions[i].init2:
                return self._solutions[i].id
        return 0

    def _init_experimental_data_graphic(self):
        # if not plt.get_fignums():  # graphics closed case

        data = np.loadtxt(self._data.experimental_data, dtype=float)
        x, y, yi = data[:, 0], data[:, 1], data[:, 2]
        Y = yi / (x * 2 * np.pi * 8.85e-12 * 9e9)

        init_tau = 1 / (2 * np.pi * x[np.argmax(Y)])
        init_eps = np.max(y) - np.min(y)

        self._init_entry_tau_k.set(f'{init_tau:.5g}')
        self._init_entry_epsilon.set(f'{init_eps:4g}')

        self._ax1.semilogx(x, Y, 'r-.', label='ε" мнимая часть диэлектрической пронициаемости', color='red')
        self._ax1.semilogx(x, y, 'r-.', label="ε' действительная часть диэлектрической пронициаемости", color='blue')

        self._ax1.set_xlabel("Гц")
        self._ax1.grid(True)
        self._ax1.set_title('Результаты эксперимента по измерению диэлектрической проницаемости')
        self._ax1.legend(loc='best', fancybox=True, shadow=False)

        self._ax2.semilogx(x, y, 'r-.', label='Экспериментальные данные')
        self._ax2.set_xlabel("f")
        self._ax2.grid(True)
        self._ax2.set_title('Диэлектрическая проницаемость')
        self._ax2.legend(loc='best', fancybox=True, shadow=False)

        # ax.set_title('real part')

        self._ax3.loglog(x, yi, 'r-.', label='Экспериментальные данные')
        self._ax3.set_xlabel("Гц")
        self._ax3.set_ylabel(r'с$^{-1}$', size=18)
        self._ax3.grid(True)
        self._ax3.set_title('Электропроводность')
        self._ax3.legend(loc='best', fancybox=True, shadow=False)

        self._figure1.tight_layout()
        self._figure2.tight_layout()
        self._figure3.tight_layout()

        self._canvas1.draw()
        self._canvas2.draw()
        self._canvas3.draw()

    def _draw_plot_to_canvas(self, s: Solutions):
        extend = 8.85e-12 * s.x * 2 * np.pi * 9e9
        self._ax2.semilogx(s.x, s.yn + s.yst, label=f'Расчетные данные № {s.id}')
        self._ax3.loglog(s.x, s.yin * extend, label=f'Расчетные данные № {s.id}')

        self._ax2.legend()
        self._ax3.legend()

        self._figure2.tight_layout()
        self._figure3.tight_layout()

        self._canvas2.draw()
        self._canvas3.draw()

    def _find_solution(self, desired_id):
        solution = None
        for i in range(len(self._solutions)):
            if self._solutions[i].id == desired_id:
                solution = self._solutions[i]

        assert solution
        return solution

    def _save_disp(self):
        desired_id = int(self._select_solution_combobox.get())
        solution = self._find_solution(desired_id)

        header = disp_header.format(layer_index=self._data.layer, poles_number=solution.poles)

        fname = os.path.join(self._path, 'dispersion')
        s = np.column_stack((solution.ak, solution.ck))
        np.savetxt(fname, s, header=header, fmt=('%.6E'), delimiter='\t', comments='')

        mb.showinfo('SAVE', f'Сохранено в {fname}')

    def __on_select_combobox_value(self, event):
        """
        Функция переключает контекс интерфейса в соответствии с выбранным решением.
        Перегружает контекст из структуры Solutions в структуру Data.
        Перезапись оосуществляется с помощью перегрузки изменяемых полей:
            _poles_entry_value
            _init_entry_value1
            _init_entry_value2
        """

        desired_id = int(self._select_solution_combobox.get())
        solution = self._find_solution(desired_id)

        self._poles_entry_value.set(str(solution.poles))
        self._init_entry_tau_k.set(f'{solution.tau:.4g}')
        self._init_entry_epsilon.set(f'{solution.epsilon:.4g}')

    def _change_file_deconstructor(self):
        self._solutions = []
        self._solutions_ids = []
        self._select_solution_combobox['value'] = [value for value in self._solutions_ids]
        self._select_solution_combobox.set('')

        self._figure1.clf()
        self._figure2.clf()
        self._figure3.clf()

        self._canvas1.draw()
        self._canvas2.draw()
        self._canvas3.draw()

        self._save_button_control(False)

    def _clear_graph_and_solutions(self):
        self._change_file_deconstructor()

        self._ax1 = self._figure1.add_subplot(111)
        self._ax2 = self._figure2.add_subplot(111)
        self._ax3 = self._figure3.add_subplot(111)

        self._save_button_control(False)
        self._init_experimental_data_graphic()
        self._dispersion_file_name_button['command'] = lambda: self._choice_file(False)
        self._clear_graph['state'] = 'disabled'

    def _change_y_log(self, item: plt.Axes, state: bool, canvas: FigureCanvasTkAgg):
        if state:
            item.set_yscale('log')
        else:
            item.set_yscale('linear')
        canvas.draw()

    def _change_x_log(self, item: plt.Axes, state: bool, canvas: FigureCanvasTkAgg):
        if state:
            item.set_xscale('log')
        else:
            item.set_xscale('linear')
        canvas.draw()

    def _activate_log_graph_changer(self):
        for val in (self.rb1_x, self.rb1_y, self.rb2_x, self.rb2_y, self.rb3_x, self.rb3_y):
            val['state'] = 'normal'


if __name__ == '__main__':
    root = tk.Tk()
    pad = 0
    root.geometry("{0}x{1}+0+0".format(
        root.winfo_screenwidth() - pad, root.winfo_screenheight() - pad))
    root.title('Дисперсия диэлектрической пронициаемости')
    # projectfilename = r'D:\test_projects\PROJECT_1_particles_die'
    try:
        print(f'Проект {projectfilename}')
        ini = os.path.normpath(projectfilename)
        ini = os.path.dirname(ini)
    except Exception:
        ini = os.getenv('USERPROFILE')

    ex = MainWindow(root, ini)
    # fp = r'C:\Users\Admin\Dropbox\work_cloud\Single_scripts\rf\ab_eps(f)_d.txt'
    # ex._choice_file(False, fp)
    root.mainloop()
