import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Optional

from src import registry
from src.registration import RawDataCapturer, EmbeddingBuilder, TargetRegistrar

logger = logging.getLogger(__name__)


class PersonRegistrySelector:
    """
    GUI shown before Stage 2 ('main.py --mode run') to pick WHICH registered person to follow
    this session, plus basic CRUD (rename/delete/register new) on the registry.

    Integrated State Flow:
    STATE_LIST -> STATE_CAPTURING (Phase 1) -> STATE_BUILDING (Phase 2 background thread) -> STATE_LIST (reloaded)
    """
    def __init__(self, config: dict, config_path: str = "config/settings.yaml"):
        self.config = config
        self.config_path = config_path
        self.selected_path: Optional[str] = None
        self.root = None
        self.tree = None

    def run(self) -> Optional[str]:
        """Returns the selected person's .npz path, or None if nothing was selected."""
        entries = registry.list_registry()

        if not entries:
            logger.info("Person registry is empty — redirecting to new registration...")
            self._register_new_blocking()
            entries = registry.list_registry()
            if not entries:
                logger.warning("Registry is still empty after registration attempt — nothing to select.")
                return None

        self._show_list_and_wait(entries)
        return self.selected_path

    def _register_new_blocking(self) -> None:
        """
        Runs Phase 1 (RawDataCapturer GUI) then Phase 2 (EmbeddingBuilder) in a background thread.
        Errors are surfaced to the user via messagebox.
        """
        try:
            capturer = RawDataCapturer(self.config)
            sanitized_name = capturer.run()
            if not sanitized_name:
                return

            self._run_embedding_builder_in_thread(sanitized_name)
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            messagebox.showerror("Registration Failed", str(e))

    def _run_embedding_builder_in_thread(self, person_name: str) -> None:
        """
        Runs EmbeddingBuilder.build_embedding() in a separate background thread so Tkinter mainloop
        is not blocked, displaying a progress dialog.
        """
        build_win = None
        if self.root and self.root.winfo_exists():
            build_win = tk.Toplevel(self.root)
            build_win.title("Building Embedding...")
            build_win.geometry("380x140")
            build_win.configure(bg="#1e1e2e")
            build_win.transient(self.root)
            build_win.grab_set()

            tk.Label(
                build_win,
                text=f"⏳ Đang build embedding cho '{person_name}'...\nVui lòng chờ trong giây lát.",
                font=("Segoe UI", 11, "bold"),
                fg="#cba6f7",
                bg="#1e1e2e",
                pady=15
            ).pack()

            pb = ttk.Progressbar(build_win, mode="indeterminate", length=280)
            pb.pack(pady=10)
            pb.start(15)

        builder_exception = [None]
        result_path = [None]

        def worker():
            try:
                builder = EmbeddingBuilder(self.config)
                result_path[0] = builder.build_embedding(person_name)
            except Exception as e:
                builder_exception[0] = e

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        # Wait for worker thread to complete while keeping Tk GUI responsive
        while thread.is_alive():
            if build_win and build_win.winfo_exists():
                build_win.update()
            elif self.root and self.root.winfo_exists():
                self.root.update()
            thread.join(timeout=0.05)

        if build_win and build_win.winfo_exists():
            build_win.destroy()

        if builder_exception[0] is not None:
            err = builder_exception[0]
            logger.error(f"Embedding build failed for '{person_name}': {err}")
            messagebox.showerror("Build Embedding Failed", str(err))
        else:
            logger.info(f"Embedding build completed: '{result_path[0]}'")
            messagebox.showinfo(
                "Build Complete",
                f"✅ Đã build embedding thành công cho '{person_name}'!\nFile: '{result_path[0]}'"
            )

    def _show_list_and_wait(self, entries) -> None:
        self.root = tk.Tk()
        self.root.title("Chọn người để theo dõi (Person Registry)")
        self.root.geometry("680x420")
        self.root.minsize(560, 340)
        self.root.configure(bg="#1e1e2e")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#313244", fieldbackground="#313244", foreground="#cdd6f4", rowheight=26)
        style.configure("Treeview.Heading", background="#181825", foreground="#cba6f7", font=("Segoe UI", 9, "bold"))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), background="#89b4fa", foreground="#11111b")

        header = tk.Label(
            self.root, text="🧑 CHỌN NGƯỜI ĐỂ THEO DÕI",
            font=("Segoe UI", 13, "bold"), fg="#cba6f7", bg="#1e1e2e",
        )
        header.pack(pady=(14, 8))

        columns = ("name", "created_at", "sample_count")
        self.tree = ttk.Treeview(self.root, columns=columns, show="headings", height=10, selectmode="browse")
        self.tree.heading("name", text="Tên")
        self.tree.heading("created_at", text="Đăng ký lúc")
        self.tree.heading("sample_count", text="Số mẫu")
        self.tree.column("name", width=240)
        self.tree.column("created_at", width=260)
        self.tree.column("sample_count", width=90, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)
        self.tree.bind("<Double-1>", lambda e: self._on_select())

        self._populate_tree(entries)

        btn_frame = tk.Frame(self.root, bg="#1e1e2e")
        btn_frame.pack(fill=tk.X, padx=14, pady=(0, 14))

        ttk.Button(btn_frame, text="▶ Select & Run", command=self._on_select).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="✏ Rename", command=self._on_rename).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="🗑 Delete", command=self._on_delete).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="➕ Register New", command=self._on_register_new).pack(side=tk.LEFT, padx=4)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _populate_tree(self, entries=None) -> None:
        if entries is None:
            entries = registry.list_registry()
        if self.tree is not None and self.tree.winfo_exists():
            for item in self.tree.get_children():
                self.tree.delete(item)
            for entry in entries:
                self.tree.insert("", tk.END, iid=entry["name"], values=(entry["name"], entry["created_at"], entry["sample_count"]))

    def _selected_name(self) -> Optional[str]:
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Vui lòng chọn 1 người trong danh sách trước.")
            return None
        return selection[0]

    def _on_select(self) -> None:
        name = self._selected_name()
        if name is None:
            return
        self.selected_path = registry.registry_path(name)
        self.root.destroy()

    def _on_delete(self) -> None:
        name = self._selected_name()
        if name is None:
            return
        if not messagebox.askyesno("Xác nhận xóa", f"Bạn có chắc muốn xóa '{name}' khỏi registry?"):
            return
        try:
            registry.delete_person(name)
            self._populate_tree()
            logger.info(f"Deleted registry entry '{name}'.")
        except Exception as e:
            messagebox.showerror("Delete Failed", str(e))

    def _on_rename(self) -> None:
        name = self._selected_name()
        if name is None:
            return
        new_name = simpledialog.askstring("Đổi tên", f"Tên mới cho '{name}':", parent=self.root)
        if new_name is None:
            return  # user cancelled
        try:
            registry.rename_person(name, new_name)
            self._populate_tree()
            logger.info(f"Renamed registry entry '{name}' -> '{new_name}'.")
        except Exception as e:
            messagebox.showerror("Rename Failed", str(e))

    def _on_register_new(self) -> None:
        self._register_new_blocking()
        self._populate_tree()

    def _on_close(self) -> None:
        self.root.destroy()
