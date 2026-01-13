import { Link, useLocation } from "react-router-dom";
import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface NavLinkItemProps {
  to: string;
  icon: LucideIcon;
  children: React.ReactNode;
  onClick?: () => void;
}

const NavLinkItem = ({ to, icon: Icon, children, onClick }: NavLinkItemProps) => {
  const location = useLocation();
  const isActive = location.pathname === to || 
    (to.startsWith('/stock/') && location.pathname.startsWith('/stock/'));

  return (
    <Link
      to={to}
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all",
        isActive
          ? "bg-primary/10 text-primary"
          : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
      )}
    >
      <Icon className="w-4 h-4" />
      {children}
    </Link>
  );
};

export default NavLinkItem;
