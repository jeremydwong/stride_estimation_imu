function varargout = matchaxes(varargin);
	if nargin > 0 ;
			h = [varargin{:}];
	else
			h = gcf;
	end

	% make all the axes match
	hallax = findobj(h,'type','axes','-not','Tag','legend','-not','Tag','Colorbar'); % stole this from "linkaxes"
	% don't include the Legend
	try
			hallax(hallax== findobj(hallax,'-depth',0,'Tag','legend') ) = [];
	end
	% don't include Empty axes (where all the Y's are NaN)
	hlines = arrayfun(@(a) findobj(a,'-depth',1,'type','line'), hallax, 'UniformOutput',0);
	yisnan = cellfun(@(b) all( arrayfun(@(a) all(isnan(get(a,'YData'))), b) ), hlines);
	hallax(yisnan) = [];
			
	% Now that we have only real data lines in hlines, find out what the
	% right limits should be:
	xmin = nanmin(cellfun(@(b) nanmin( arrayfun(@(a) nanmin(get(a,'XData')), b) ), hlines));
	xmax = nanmax(cellfun(@(b) nanmax( arrayfun(@(a) nanmax( get(a,'XData')), b) ), hlines));
	ymin = nanmin(cellfun(@(b) nanmin( arrayfun(@(a) nanmin(get(a,'YData')), b) ), hlines));
	ymax = nanmax(cellfun(@(b) nanmax( arrayfun(@(a) nanmax( get(a,'YData')), b) ), hlines));

	try
		zmin = nanmin(cellfun(@(b) nanmin( arrayfun(@(a) nanmin(get(a,'ZData')), b) ), hlines));
		zmax = nanmax(cellfun(@(b) nanmax( arrayfun(@(a) nanmax( get(a,'ZData')), b) ), hlines));
	catch
			zmin = -1; zmax = 1;
	end

	set(hallax,'XLim',[xmin xmax],'YLim',[ymin ymax],'ZLim',[zmin zmax]);
	hlink = linkprop(hallax,{'XLim','YLim','ZLim','View'}) ;

	set(h,'UserData',{get(h,'UserData'),hlink});

	if nargout == 1
			varargout{1} = hlink; 
	end

