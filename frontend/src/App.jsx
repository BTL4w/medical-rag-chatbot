import React, { useState } from "react";
import Login from "./components/Login";
import Register from "./components/Register";
import Chat from "./components/Chat";
import "./App.css";

function App() {
  const [view, setView] = useState("login");
  const [isAuthenticated, setIsAuthenticated] = useState(
    !!localStorage.getItem("session_id")
  );

  if (isAuthenticated) {
    return (
      <Chat
        onLogout={() => {
          setIsAuthenticated(false);
          setView("login");
        }}
      />
    );
  }

  return (
    <div className="App">
      {view === "login" && (
        <>
          <Login onLogin={() => setIsAuthenticated(true)} />
          <p>
            Don't have an account?{" "}
            <button onClick={() => setView("register")}>Register</button>
          </p>
        </>
      )}

      {view === "register" && (
        <>
          <Register onRegister={() => setView("login")} />
          <p>
            Already have an account?{" "}
            <button onClick={() => setView("login")}>Login</button>
          </p>
        </>
      )}
    </div>
  );
}

export default App;
