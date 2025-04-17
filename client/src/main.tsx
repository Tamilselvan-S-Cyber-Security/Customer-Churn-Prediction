import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

// Set document title
document.title = "CSV Data Processor";

createRoot(document.getElementById("root")!).render(<App />);
