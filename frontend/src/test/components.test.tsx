import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { AuthProvider } from "../auth/AuthContext";
import { Meter, statusTone } from "../components/ui";
import LoginPage from "../pages/Login";

describe("ui components", () => {
  it("meter reports its value and caps at 100%", () => {
    render(<Meter ratio={1.4} threshold={0.8} />);
    expect(screen.getByRole("meter")).toHaveAttribute("aria-valuenow", "100");
  });

  it("maps statuses to tones", () => {
    expect(statusTone("success")).toBe("green");
    expect(statusTone("soft_alert")).toBe("yellow");
    expect(statusTone("blocked_budget")).toBe("red");
    expect(statusTone("something_else")).toBe("gray");
  });
});

describe("login page", () => {
  beforeEach(() => localStorage.clear());

  it("shows the server's error on bad credentials", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Incorrect email or password" }), { status: 401 }));
    render(
      <QueryClientProvider client={new QueryClient()}>
        <AuthProvider><LoginPage /></AuthProvider>
      </QueryClientProvider>,
    );
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "x@example.com" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "wrong-pass" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent("Incorrect email or password"));
  });
});
