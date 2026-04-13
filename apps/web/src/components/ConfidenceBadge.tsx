import type { RiskLevel } from "@/lib/api";

interface Props {
  confidence: number;
  riskLevel: RiskLevel;
}

const RISK_CONFIG: Record<RiskLevel, { label: string; className: string }> = {
  LOW:      { label: "Low Risk",      className: "bg-green-100 text-green-800 border-green-200" },
  MEDIUM:   { label: "Medium Risk",   className: "bg-yellow-100 text-yellow-800 border-yellow-200" },
  HIGH:     { label: "High Risk",     className: "bg-orange-100 text-orange-800 border-orange-200" },
  CRITICAL: { label: "Critical Risk", className: "bg-red-100 text-red-800 border-red-200" },
};

export function ConfidenceBadge({ confidence, riskLevel }: Props) {
  const { label, className } = RISK_CONFIG[riskLevel] ?? RISK_CONFIG.HIGH;
  const pct = Math.round(confidence * 100);

  return (
    <div className="flex items-center gap-3">
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${className}`}>
        {label}
      </span>
      <div className="flex items-center gap-1.5">
        <div className="w-24 h-1.5 rounded-full bg-gray-200 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${
              pct >= 75 ? "bg-green-500" :
              pct >= 50 ? "bg-yellow-500" :
              pct >= 30 ? "bg-orange-500" : "bg-red-500"
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-xs text-gray-500 tabular-nums">{pct}%</span>
      </div>
    </div>
  );
}
