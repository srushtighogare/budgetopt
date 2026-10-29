import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

// Model answers are Markdown (headings, bold, lists, GitHub-style tables). Raw HTML in an
// answer is not rendered: react-markdown escapes it, so a reply cannot inject markup.
const components: Components = {
  h1: ({ children }) => <h3 className="mt-5 mb-2 text-base font-semibold text-slate-900 first:mt-0">{children}</h3>,
  h2: ({ children }) => <h3 className="mt-5 mb-2 text-base font-semibold text-slate-900 first:mt-0">{children}</h3>,
  h3: ({ children }) => <h4 className="mt-4 mb-1.5 text-sm font-semibold text-slate-900 first:mt-0">{children}</h4>,
  h4: ({ children }) => <h5 className="mt-3 mb-1 text-sm font-semibold text-slate-800 first:mt-0">{children}</h5>,
  p: ({ children }) => <p className="my-2 first:mt-0 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="my-2 list-disc space-y-1 pl-5">{children}</ul>,
  ol: ({ children }) => <ol className="my-2 list-decimal space-y-1 pl-5">{children}</ol>,
  strong: ({ children }) => <strong className="font-semibold text-slate-900">{children}</strong>,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer noopener" className="text-brand-700 underline">{children}</a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-3 border-l-2 border-slate-300 pl-3 text-slate-600">{children}</blockquote>
  ),
  hr: () => <hr className="my-4 border-slate-200" />,
  code: ({ className, children }) =>
    className ? <code className={className}>{children}</code>
      : <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[0.85em] text-slate-800">{children}</code>,
  pre: ({ children }) => (
    <pre className="my-3 overflow-x-auto rounded-lg bg-slate-900 p-3 font-mono text-xs leading-relaxed text-slate-100">{children}</pre>
  ),
  table: ({ children }) => (
    <div className="my-3 overflow-x-auto rounded-lg border border-slate-200">
      <table className="w-full border-collapse text-left text-[13px]">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-slate-50">{children}</thead>,
  th: ({ children }) => <th className="border-b border-slate-200 px-3 py-2 font-semibold text-slate-700">{children}</th>,
  td: ({ children }) => <td className="border-b border-slate-100 px-3 py-2 align-top">{children}</td>,
};

// Models put <br> inside table cells for line breaks. That one tag becomes a real break;
// every other raw HTML node is still escaped by react-markdown.
type HastNode = { type: string; value?: string; tagName?: string; properties?: object;
                  children?: HastNode[] };

function brToBreak(node: HastNode) {
  node.children?.forEach((child, i, siblings) => {
    if (child.type === "raw" && /^<br\s*\/?>$/i.test(child.value?.trim() ?? "")) {
      siblings[i] = { type: "element", tagName: "br", properties: {}, children: [] };
    } else {
      brToBreak(child);
    }
  });
}

const rehypeBr = () => (tree: HastNode) => brToBreak(tree);

export function Markdown({ children }: { children: string }) {
  return (
    <div className="text-sm leading-relaxed text-slate-800">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeBr]}
                     components={components}>{children}</ReactMarkdown>
    </div>
  );
}
