import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider, createGlobalStyle } from "styled-components";
import { theme } from "./theme";
import ContactList from "./components/ContactList";
import CallScreen from "./components/CallScreen";
import SpeakerCardEditor from "./components/SpeakerCardEditor";
import ProcessingView from "./components/ProcessingView";

const GlobalStyle = createGlobalStyle`
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: ${(p) => p.theme.colors.paper};
    color: ${(p) => p.theme.colors.ink};
    font-family: ${(p) => p.theme.font.body};
    font-feature-settings: "tnum";           /* 타이머·수치 고정폭 숫자 */
    -webkit-font-smoothing: antialiased;
  }
  ::selection { background: ${(p) => p.theme.colors.accent}; color: ${(p) => p.theme.colors.onAccent}; }
  input, textarea, select, button { font-family: inherit; }
  :focus-visible { outline: 2px solid ${(p) => p.theme.colors.ink}; outline-offset: 2px; }
`;

export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<ContactList />} />
          <Route path="/call/:id" element={<CallScreen />} />
          <Route path="/editor/:id" element={<SpeakerCardEditor />} />
          <Route path="/processing" element={<ProcessingView />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}
