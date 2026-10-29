import { render, screen } from "@testing-library/react";

import { Markdown } from "../components/Markdown";

describe("Markdown answers", () => {
  it("renders headings, bold text and GitHub-style tables", () => {
    const { container } = render(<Markdown>{[
      "## Quick Take",
      "**Recommendation:** start with a monolith.",
      "",
      "| Feature | Monolith |",
      "|---|---|",
      "| Deployment | One unit |",
    ].join("\n")}</Markdown>);
    expect(screen.getByRole("heading", { name: "Quick Take" })).toBeInTheDocument();
    expect(screen.getByText("Recommendation:").tagName).toBe("STRONG");
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "One unit" })).toBeInTheDocument();
    expect(container.textContent).not.toContain("**");
  });

  it("turns <br> in a table cell into a line break", () => {
    const { container } = render(<Markdown>{"| A | B |\n|---|---|\n| x | one<br>two |"}</Markdown>);
    expect(container.querySelector("td br")).not.toBeNull();
    expect(container.textContent).not.toContain("<br>");
  });

  it("does not render raw HTML from a model answer", () => {
    const { container } = render(<Markdown>{'Hi <img src=x onerror="alert(1)"> there'}</Markdown>);
    expect(container.querySelector("img")).toBeNull();
  });
});
